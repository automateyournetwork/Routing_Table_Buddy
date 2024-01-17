import os
import json
import logging
from pyats import aetest
from pyats.log.utils import banner

# ----------------
# Get logger for script
# ----------------

log = logging.getLogger(__name__)

# ----------------
# AE Test Setup
# ----------------
class common_setup(aetest.CommonSetup):
    """Common Setup section"""
# ----------------
# Connected to devices
# ----------------
    @aetest.subsection
    def connect_to_devices(self, testbed):
        """Connect to all the devices"""
        testbed.connect()
# ----------------
# Mark the loop for Learn Interfaces
# ----------------
    @aetest.subsection
    def loop_mark(self, testbed):
        aetest.loop.mark(Show_IP_Route_Langchain, device_name=testbed.devices)

# ----------------
# Test Case #1
# ----------------
class Show_IP_Route_Langchain(aetest.Testcase):
    """pyATS Get and Save Show IP Route"""

    @aetest.test
    def setup(self, testbed, device_name):
        """ Testcase Setup section """
        # Set current device in loop as self.device
        self.device = testbed.devices[device_name]

    @aetest.test
    def get_raw_config(self):
        raw_json = self.device.parse("show ip route")
        
        self.parsed_json = {"info": raw_json}

    @aetest.test
    def create_file(self):
        with open('Show_IP_Route.json', 'w') as f:
            f.write(json.dumps(self.parsed_json, indent=4, sort_keys=True))

class CommonCleanup(aetest.CommonCleanup):
    @aetest.subsection
    def disconnect_from_devices(self, testbed):
        testbed.disconnect()

# for running as its own executable
if __name__ == '__main__':
    aetest.main()