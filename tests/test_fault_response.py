import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simulation.physics import PhysicalSimulator
from controllers.engine import CyberPhysicalController


class TestFaultResponse(unittest.TestCase):

    def test_temperature_sensor_drift_is_observable(self):
        sim = PhysicalSimulator()

        sim.temp_sensor_drift = 5.0
        sensors = sim.read_sensors()

        self.assertTrue(
            math.isclose(
                sensors["temperature_k"],
                sim.true_core_temp_k + 5.0,
                abs_tol=0.10,
            )
        )

    def test_harvester_fault_changes_sensor_contract(self):
        sim = PhysicalSimulator()

        self.assertTrue(sim.read_sensors()["harvester_active"])

        sim.microcracks_occurred = True
        sim.true_harvester_efficiency = 0.4

        sensors = sim.read_sensors()

        self.assertFalse(sensors["harvester_active"])

    def test_controller_enters_survival_on_low_soc(self):
        ctl = CyberPhysicalController()

        ctl.estimated_soc = 0.39
        ctl.last_temp_reading = 298.15
        ctl.temp_slope = 0.0

        actuators = ctl.coordinate(1.0)

        self.assertEqual(ctl.current_mode, "SURVIVAL")
        self.assertEqual(actuators["cpu_hz"], 8_000_000)
        self.assertEqual(actuators["tx_power_dbm"], -10.0)

    def test_controller_enters_recovery_on_critical_soc(self):
        ctl = CyberPhysicalController()

        ctl.estimated_soc = 0.19
        ctl.last_temp_reading = 298.15
        ctl.temp_slope = 0.0

        actuators = ctl.coordinate(1.0)

        self.assertEqual(ctl.current_mode, "RECOVERY")
        self.assertEqual(actuators["cpu_hz"], 1_000_000)
        self.assertEqual(actuators["tx_power_dbm"], -40.0)

    def test_controller_enters_survival_on_predicted_temperature(self):
        ctl = CyberPhysicalController()

        ctl.estimated_soc = 0.80
        ctl.last_temp_reading = 310.15
        ctl.temp_slope = 0.10

        actuators = ctl.coordinate(1.0)

        self.assertEqual(ctl.current_mode, "SURVIVAL")
        self.assertEqual(actuators["cpu_hz"], 8_000_000)
        self.assertEqual(actuators["tx_power_dbm"], -10.0)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestFaultResponse)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
