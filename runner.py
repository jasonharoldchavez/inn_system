import json
import math
import socket
import time

from simulation.physics import PhysicalSimulator
from controllers.engine import CyberPhysicalController


def run_loop(
    cycles: int | None = None,
    realtime: bool = True,
    acceleration: float = 100.0,
    broadcast: bool = False,
):
    simulator = PhysicalSimulator()
    controller = CyberPhysicalController()

    sock = None
    addr = ("<broadcast>", 5005)

    if broadcast:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

    dt = 1.0
    simulated_hours = 0.0
    cycle = 0

    print("\n=======================================================")
    print(" INN CYBER-PHYSICAL NODE SIMULATION")
    print(" Hardware-in-the-loop style execution")
    print(f" Acceleration: {acceleration}x")
    print(f" Cycles: {'continuous' if cycles is None else cycles}")
    print("=======================================================\n")

    try:
        while cycles is None or cycle < cycles:
            cycle += 1

            accelerated_dt = dt * acceleration
            simulated_hours += accelerated_dt / 3600.0

            simulator.ambient_temp_k = (
                298.15 + 15.0 * math.sin(simulated_hours * 0.26)
            )

            actuators = controller.coordinate(accelerated_dt)

            simulator.step(
                accelerated_dt,
                actuators["cpu_hz"],
                actuators["tx_power_dbm"],
            )

            sensors = simulator.read_sensors()

            controller.update_estimators(
                sensors,
                accelerated_dt,
            )

            battery_health = (
                simulator.true_battery_capacity_as / 3600.0
            ) * 100.0

            print(
                f"Cycle: {cycle:5d} | "
                f"SimHour: {simulated_hours:7.2f} | "
                f"Mode: {controller.current_mode:8} | "
                f"Capacity: {simulator.true_battery_capacity_as:8.2f}s | "
                f"Health: {battery_health:6.2f}% | "
                f"SoC: {controller.estimated_soc * 100:6.2f}% | "
                f"Temp: {simulator.true_core_temp_k - 273.15:6.2f}C | "
                f"Sensor: {sensors['temperature_k'] - 273.15:6.2f}C | "
                f"Drift: {simulator.temp_sensor_drift:7.3f}K | "
                f"Harvest: {'OK' if sensors['harvester_active'] else 'DEGRADED'}"
            )

            payload = {
                "cycle": cycle,
                "simulated_hours": simulated_hours,
                "mode": controller.current_mode,
                "battery_capacity_as": simulator.true_battery_capacity_as,
                "battery_health_pct": battery_health,
                "estimated_soc": controller.estimated_soc,
                "true_temperature_k": simulator.true_core_temp_k,
                "sensor_temperature_k": sensors["temperature_k"],
                "temperature_sensor_drift_k": simulator.temp_sensor_drift,
                "harvester_active": sensors["harvester_active"],
                "actuators": actuators,
            }

            if sock is not None:
                try:
                    sock.sendto(
                        json.dumps(payload).encode(),
                        addr,
                    )
                except OSError:
                    pass

            if realtime:
                time.sleep(dt)

    except KeyboardInterrupt:
        print("\n[+] Simulation halted safely by user.")

    finally:
        if sock is not None:
            sock.close()

    return simulator, controller


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="INN cyber-physical HIL simulation"
    )

    parser.add_argument(
        "--cycles",
        type=int,
        default=None,
        help="Number of cycles to execute; omit for continuous mode",
    )

    parser.add_argument(
        "--acceleration",
        type=float,
        default=100.0,
        help="Simulation-time acceleration factor",
    )

    parser.add_argument(
        "--no-realtime",
        action="store_true",
        help="Run without one-second wall-clock pacing",
    )

    parser.add_argument(
        "--broadcast",
        action="store_true",
        help="Enable UDP broadcast telemetry on port 5005",
    )

    args = parser.parse_args()

    run_loop(
        cycles=args.cycles,
        realtime=not args.no_realtime,
        acceleration=args.acceleration,
        broadcast=args.broadcast,
    )
