import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simulation.physics import PhysicalSimulator
from controllers.engine import CyberPhysicalController


class TestEnergyAccounting(unittest.TestCase):

    def test_physics_tracks_energy_throughput(self):
        sim = PhysicalSimulator()

        before = sim.battery_throughput_ah

        sim.step(
            dt_sec=100.0,
            cpu_freq_hz=120_000_000,
            tx_power_dbm=14.0,
        )

        self.assertGreater(sim.battery_throughput_ah, before)
        self.assertTrue(math.isfinite(sim.battery_throughput_ah))

    def test_capacity_changes_under_sustained_load(self):
        sim = PhysicalSimulator()

        before = sim.true_battery_capacity_as

        for _ in range(100):
            sim.step(
                dt_sec=100.0,
                cpu_freq_hz=120_000_000,
                tx_power_dbm=14.0,
            )

        self.assertLess(sim.true_battery_capacity_as, before)

    def test_controller_soc_is_not_independent_of_physical_state(self):
        sim = PhysicalSimulator()
        ctl = CyberPhysicalController()

        initial_soc = ctl.estimated_soc

        for _ in range(20):
            actuators = ctl.coordinate(100.0)
            sim.step(
                100.0,
                actuators["cpu_hz"],
                actuators["tx_power_dbm"],
            )
            sensors = sim.read_sensors()
            ctl.update_estimators(sensors, 100.0)

        self.assertLess(ctl.estimated_soc, initial_soc)
        self.assertTrue(math.isfinite(ctl.estimated_soc))


class TestThermalNumericalStability(unittest.TestCase):

    def test_large_timestep_temperature_converges_without_oscillation(self):
        sim = PhysicalSimulator()
        temperatures = []

        for _ in range(100):
            sim.step(
                dt_sec=100.0,
                cpu_freq_hz=120_000_000,
                tx_power_dbm=14.0,
            )
            temperatures.append(sim.true_core_temp_k)

        self.assertTrue(
            all(math.isfinite(t) and t > 0 for t in temperatures)
        )

        # Under constant conditions, this first-order model
        # should approach equilibrium monotonically.
        differences = [
            temperatures[i + 1] - temperatures[i]
            for i in range(len(temperatures) - 1)
        ]

        self.assertTrue(
            all(d >= -1e-10 for d in differences),
            "Temperature unexpectedly oscillated or decreased.",
        )

        self.assertLess(
            abs(temperatures[-1] - temperatures[-2]),
            1e-6,
        )



    def test_thermal_step_matches_exact_first_order_solution(self):
        sim = PhysicalSimulator()

        dt = 10.0
        initial_temp = sim.true_core_temp_k
        ambient_temp = sim.ambient_temp_k

        cpu_power = 0.05
        radio_power = 0.25 * (10 ** (14.0 / 10.0)) / 10.0
        heat_generated = (cpu_power + radio_power) * 0.15
        conductance = 0.02

        equilibrium = ambient_temp + heat_generated / conductance
        expected = equilibrium + (
            initial_temp - equilibrium
        ) * math.exp(-conductance * dt)

        sim.step(dt, 120_000_000, 14.0)

        self.assertAlmostEqual(
            sim.true_core_temp_k,
            expected,
            places=10,
        )


class TestBatteryChargeIntegration(unittest.TestCase):

    def test_controller_uses_measured_battery_soc(self):
        ctl = CyberPhysicalController()

        ctl.update_estimators({
            "temperature_k": 298.15,
            "voltage_v": 3.7,
            "battery_soc": 0.42,
        }, 1.0)

        self.assertAlmostEqual(ctl.estimated_soc, 0.42)

    def test_controller_ignores_nonfinite_soc_measurement(self):
        ctl = CyberPhysicalController()
        ctl.estimated_soc = 0.65

        ctl.update_estimators({
            "temperature_k": 298.15,
            "voltage_v": 3.7,
            "battery_soc": float("nan"),
        }, 1.0)

        self.assertAlmostEqual(ctl.estimated_soc, 0.65)

    def test_energy_harvesting_can_increase_stored_charge(self):
        sim = PhysicalSimulator()
        sim.base_vibration_g = 10.0
        sim.battery_charge_as = 3000.0
        before = sim.battery_charge_as

        sim.step(
            dt_sec=10.0,
            cpu_freq_hz=1_000_000,
            tx_power_dbm=-40.0,
        )

        self.assertGreater(sim.battery_charge_as, before)
        self.assertLessEqual(
            sim.battery_charge_as,
            sim.true_battery_capacity_as,
        )

    def test_charge_and_soc_remain_within_capacity_bounds(self):
        sim = PhysicalSimulator()

        for _ in range(100):
            sim.step(100.0, 120_000_000, 14.0)

        sensors = sim.read_sensors()

        self.assertGreaterEqual(sim.battery_charge_as, 0.0)
        self.assertLessEqual(
            sim.battery_charge_as,
            sim.true_battery_capacity_as,
        )
        self.assertGreaterEqual(sensors["battery_soc"], 0.0)
        self.assertLessEqual(sensors["battery_soc"], 1.0)

if __name__ == "__main__":
    unittest.main()
