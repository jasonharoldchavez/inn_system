import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simulation.physics import PhysicalSimulator
from controllers.engine import CyberPhysicalController


class TestINNSystem(unittest.TestCase):

    def test_initial_state(self):
        sim = PhysicalSimulator()
        ctl = CyberPhysicalController()

        self.assertEqual(sim.true_battery_capacity_as, 3600.0)
        self.assertTrue(
            math.isclose(
                sim.true_core_temp_k,
                298.15,
                rel_tol=0,
                abs_tol=1e-9,
            )
        )
        self.assertEqual(ctl.estimated_soc, 1.0)
        self.assertEqual(ctl.current_mode, "NOMINAL")

    def test_physics_step_is_finite(self):
        sim = PhysicalSimulator()

        sim.step(
            dt_sec=1.0,
            cpu_freq_hz=48_000_000,
            tx_power_dbm=0.0,
        )

        self.assertTrue(math.isfinite(sim.true_core_temp_k))
        self.assertTrue(math.isfinite(sim.true_battery_capacity_as))
        self.assertTrue(math.isfinite(sim.battery_throughput_ah))

        self.assertGreaterEqual(sim.true_battery_capacity_as, 10.0)

    def test_sensor_contract(self):
        sim = PhysicalSimulator()

        sensors = sim.read_sensors()

        self.assertIn("temperature_k", sensors)
        self.assertIn("voltage_v", sensors)
        self.assertIn("harvester_active", sensors)

        self.assertTrue(math.isfinite(sensors["temperature_k"]))
        self.assertTrue(math.isfinite(sensors["voltage_v"]))
        self.assertIsInstance(sensors["harvester_active"], bool)

    def test_controller_actuator_contract(self):
        ctl = CyberPhysicalController()

        actuators = ctl.coordinate(1.0)

        self.assertIn("cpu_hz", actuators)
        self.assertIn("tx_power_dbm", actuators)

        self.assertGreater(actuators["cpu_hz"], 0)
        self.assertTrue(math.isfinite(actuators["tx_power_dbm"]))

    def test_controller_sensor_update(self):
        sim = PhysicalSimulator()
        ctl = CyberPhysicalController()

        sensors = sim.read_sensors()
        ctl.update_estimators(sensors, 1.0)

        self.assertTrue(math.isfinite(ctl.estimated_soc))
        self.assertTrue(math.isfinite(ctl.temp_slope))
        self.assertTrue(math.isfinite(ctl.kalman_r_covariance))

        self.assertGreaterEqual(ctl.estimated_soc, 0.0)
        self.assertLessEqual(ctl.estimated_soc, 1.0)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestINNSystem)
    result = unittest.TextTestRunner(verbosity=2).run(suite)

    if not result.wasSuccessful():
        sys.exit(1)
