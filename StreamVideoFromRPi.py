import paramiko
import os


class StreamVideoFromRPi:

    def __init__(self):
        self.hostname = '192.168.0.31'
        self.username = 'pi'
        private_key_file = os.path.expanduser('~/.ssh/id_rsa')
        self.my_key = paramiko.RSAKey.from_private_key_file(private_key_file, password='secretPassword')
        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.client.load_system_host_keys()

    def start_stream(self):
        self.client.connect(hostname=self.hostname, username=self.username, pkey=self.my_key)
        print('Start streaming from RPi..')
        stdin, stdout, stderr = self.client.exec_command('~/Desktop/./streamVideo.sh > /dev/null 2>&1 &')
        #stdin, stdout, stderr = self.client.exec_command('~/Desktop/./streamVideo.sh')
        self.print_logs_and_errors(stdout, stderr)
        self.client.close()

    def stop_stream(self):
        process_to_kill = 'gst-launch-1.0'
        self.client.connect(hostname=self.hostname, username=self.username, pkey=self.my_key)
        stdin, stdout, stderr = self.client.exec_command('pkill -f ' + process_to_kill)
        print('Stop streaming and kill depend processes...')
        self.print_logs_and_errors(stdout, stderr)
        self.client.close()

    def print_logs_and_errors(self, stdout, stderr):
        output = stdout.readlines()
        output_errors = stderr.readlines()

        print('\n'.join(output))
        print('\n'.join(output_errors))
