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


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        TestEnergyAccounting
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
