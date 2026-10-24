import io
import unittest
from contextlib import redirect_stdout

from station.contracts import OutputDriver, StationError
from station.outputs import LAMPS, ConsoleOutputs, RecordingOutputs, SerialRelayOutputs

OFF = dict.fromkeys(LAMPS, False)
ACTIONS = ["show_good", "show_bad", "alarm_off", "show_review", "show_fault", "clear",
           "show_bad", "show_good"]


def expected_after(action, prev):
    """The invariant table from the module docstring, as full lamp states."""
    if action == "show_good":
        return dict(OFF, green=True)
    if action == "show_bad":
        return dict(OFF, red=True, alarm=True)
    if action == "show_review":
        return dict(OFF, amber=True)
    if action == "show_fault":
        return dict(OFF, fault=True)
    if action == "alarm_off":
        return dict(prev, alarm=False)
    if action == "clear":
        return dict(OFF)
    raise AssertionError(action)


def run_sequence(test, driver, observe):
    """Drive ACTIONS through driver; observe(driver) gives the lamp dict after each step."""
    prev = dict(OFF)
    for action in ACTIONS:
        getattr(driver, action)()
        state = observe()
        test.assertEqual(state, expected_after(action, prev), f"after {action}")
        prev = state


class InvariantTableTests(unittest.TestCase):
    """Spot checks of the rules the spec calls out explicitly."""

    def test_good_never_lights_red_or_alarm(self):
        s = expected_after("show_good", OFF)
        self.assertTrue(s["green"])
        self.assertFalse(s["red"] or s["alarm"])

    def test_bad_is_red_and_alarm_never_green(self):
        s = expected_after("show_bad", OFF)
        self.assertTrue(s["red"] and s["alarm"])
        self.assertFalse(s["green"])

    def test_review_is_amber_only(self):
        s = expected_after("show_review", OFF)
        self.assertEqual({k for k, v in s.items() if v}, {"amber"})

    def test_fault_is_fault_only(self):
        s = expected_after("show_fault", OFF)
        self.assertEqual({k for k, v in s.items() if v}, {"fault"})


class RecordingOutputsTests(unittest.TestCase):
    def test_implements_output_driver_protocol_shape(self):
        driver: OutputDriver = RecordingOutputs()
        for name in ("show_good", "show_bad", "show_review", "show_fault", "alarm_off", "clear"):
            self.assertTrue(callable(getattr(driver, name)))

    def test_initial_state_all_off_and_empty_history(self):
        driver = RecordingOutputs()
        self.assertEqual(observe_attrs(driver), OFF)
        self.assertEqual(driver.history, [])

    def test_invariants_over_sequence(self):
        driver = RecordingOutputs()
        run_sequence(self, driver, lambda: observe_attrs(driver))

    def test_history_records_calls_in_order(self):
        driver = RecordingOutputs()
        driver.show_good()
        driver.clear()
        self.assertEqual(driver.history, ["show_good", "clear"])

    def test_alarm_off_leaves_red_on(self):
        driver = RecordingOutputs()
        driver.show_bad()
        driver.alarm_off()
        self.assertTrue(driver.red)
        self.assertFalse(driver.alarm)


def observe_attrs(driver):
    return {lamp: getattr(driver, lamp) for lamp in LAMPS}


class ConsoleOutputsTests(unittest.TestCase):
    def test_invariants_over_sequence_via_printed_lines(self):
        lines = []
        driver = ConsoleOutputs(write=lines.append)

        def observe():
            # The most recent line carries the full state after the call.
            return parse_console_line(lines[-1])

        run_sequence(self, driver, observe)
        self.assertEqual(len(lines), len(ACTIONS))

    def test_one_line_per_call_with_action_name(self):
        lines = []
        driver = ConsoleOutputs(write=lines.append)
        driver.show_bad()
        driver.alarm_off()
        self.assertEqual(len(lines), 2)
        self.assertTrue(lines[0].startswith("OUTPUTS show_bad:"))
        self.assertTrue(lines[1].startswith("OUTPUTS alarm_off:"))
        self.assertNotIn("\n", lines[0])

    def test_default_writer_is_print(self):
        buf = io.StringIO()
        driver = ConsoleOutputs()
        with redirect_stdout(buf):
            driver.show_fault()
        self.assertEqual(
            buf.getvalue().strip(),
            "OUTPUTS show_fault: green=OFF red=OFF alarm=OFF amber=OFF fault=ON",
        )


def parse_console_line(line):
    _, _, rest = line.partition(": ")
    state = {}
    for token in rest.split():
        lamp, _, value = token.partition("=")
        state[lamp] = value == "ON"
    assert set(state) == set(LAMPS), line
    return state


class SerialRelayOutputsTests(unittest.TestCase):
    def setUp(self):
        self.written = []
        self.channels = {"green": 1, "red": 2, "alarm": 3, "amber": 4, "fault": 5}
        self.driver = SerialRelayOutputs(self.written.append)

    def test_show_good_writes_all_five_channels(self):
        self.driver.show_good()
        self.assertEqual(
            self.written,
            [b"ON1\n", b"OFF2\n", b"OFF3\n", b"OFF4\n", b"OFF5\n"],
        )

    def test_show_bad_turns_red_and_alarm_on(self):
        self.driver.show_bad()
        self.assertEqual(self.written[:3], [b"OFF1\n", b"ON2\n", b"ON3\n"])

    def test_alarm_off_writes_only_alarm_channel(self):
        self.driver.show_bad()
        self.written.clear()
        self.driver.alarm_off()
        self.assertEqual(self.written, [b"OFF3\n"])

    def test_clear_writes_off_for_everything(self):
        self.driver.show_bad()
        self.written.clear()
        self.driver.clear()
        self.assertEqual(
            self.written,
            [b"OFF1\n", b"OFF2\n", b"OFF3\n", b"OFF4\n", b"OFF5\n"],
        )

    def test_invariants_over_sequence_by_replaying_bytes(self):
        relay = dict.fromkeys(LAMPS, False)
        inverse = {ch: lamp for lamp, ch in self.channels.items()}
        consumed = 0

        def observe():
            nonlocal consumed
            for line in self.written[consumed:]:
                self.assertTrue(line.endswith(b"\n"))
                text = line.decode("ascii").strip()
                if text.startswith("ON"):
                    on, channel = True, text[2:]
                else:
                    self.assertTrue(text.startswith("OFF"), text)
                    on, channel = False, text[3:]
                relay[inverse[int(channel)]] = on
            consumed = len(self.written)
            return dict(relay)

        run_sequence(self, self.driver, observe)

    def test_custom_channel_map(self):
        written = []
        driver = SerialRelayOutputs(written.append, channel_map={
            "green": 8, "red": 7, "alarm": 6, "amber": 5, "fault": 4})
        driver.show_good()
        self.assertEqual(written[0], b"ON8\n")
        self.assertEqual(written[1], b"OFF7\n")

    def test_incomplete_channel_map_rejected(self):
        with self.assertRaises(StationError):
            SerialRelayOutputs(lambda b: None, channel_map={"green": 1})

    def test_duplicate_channel_rejected(self):
        with self.assertRaises(StationError):
            SerialRelayOutputs(lambda b: None, channel_map={
                "green": 1, "red": 1, "alarm": 3, "amber": 4, "fault": 5})

    def test_unknown_output_rejected(self):
        with self.assertRaises(StationError):
            SerialRelayOutputs(lambda b: None, channel_map={
                "green": 1, "red": 2, "alarm": 3, "amber": 4, "fault": 5, "buzzer": 6})


if __name__ == "__main__":
    unittest.main()
