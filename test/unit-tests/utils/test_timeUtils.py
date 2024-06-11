import time
import unittest

import vvadlrs3.utils.timeUtils as tUtils
@tUtils.timeit
def sleep_x_sec(sleep_time, **kwargs):
    """
        Dedicated sleep timer to delay an operation in a time measured task

        Args:
            sleep_time (int): sleeping time in seconds
    """

    time.sleep(sleep_time)

class TestTimeUtils(unittest.TestCase):
    """
        Test the time measurement using a sleep function
    """

    def test_timeit(self):
        """
            Unit test on the timeit function. We provoke a function that is active
            for 5000 milliseconds. timeit should log this time as the expected execution
            time of the given function.
            A tolerance is needed due to the system's performance
        """
        
        logtime_data = {}
        sleep_time = 5
        sleep_x_sec(sleep_time, log_time=logtime_data)
        print(f"data is {logtime_data}")
        print("name is", sleep_x_sec.__name__)
        self.assertLessEqual(logtime_data.get("SLEEP_X_SEC"), sleep_time * 1000 * 1.01)


if __name__ == '__main__':
    unittest.main()
