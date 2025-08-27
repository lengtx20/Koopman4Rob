''' This Class reads data from record files from CyberDog '''

import numpy as np

class CyberDogDataProcessor:
    def __init__(self, file_name):
        self.file_name = file_name
        self.data = np.load(file_name, allow_pickle=True)
        self.time_step = 20000  # 20000ns = 0.02s
        self.timestamps = np.array([entry['timestamp'] for entry in self.data])

    def process_data(self):
        x_vel_des_list = []
        y_vel_des_list = []
        yaw_vel_des_list = []
        rpy_des_list = []

        vWorld_list = []
        vBody_list = []
        rpy_list = []
        omegaBody_list = []
        omegaWorld_list = []
        quat_list = []
        aBody_list = []
        aWorld_list = []

        q_list = []
        qd_list = []
        p_list = []
        v_list = []
        tau_est_list = []
        force_est_list = []

        # Find the minimum and maximum timestamps
        min_timestamp = min(self.timestamps)
        max_timestamp = max(self.timestamps)
        # Initialize the current time to the minimum timestamp
        current_time = min_timestamp

        # align timestamps to the nearest time step
        while current_time <= max_timestamp:
            # Find the index of the closest timestamp for each data channel
            motion_control_index = np.argmin(np.abs(self.timestamps - current_time))
            state_estimator_index = np.argmin(np.abs(self.timestamps - current_time))
            leg_control_data_index = np.argmin(np.abs(self.timestamps - current_time))

            # motion_control
            motion_control = self.data[motion_control_index]['motion_control']
            while motion_control is None and motion_control_index < len(self.timestamps) - 1:
                motion_control_index += 1
                motion_control = self.data[motion_control_index]['motion_control']
            if motion_control is not None:
                if 'x_vel_des' in motion_control:
                    x_vel_des_list.append(motion_control['x_vel_des'])
                if 'y_vel_des' in motion_control:
                    y_vel_des_list.append(motion_control['y_vel_des'])
                if 'yaw_vel_des' in motion_control:
                    yaw_vel_des_list.append(motion_control['yaw_vel_des'])
                if 'rpy_des' in motion_control:
                    rpy_des_list.append(motion_control['rpy_des'])

            # state_estimator
            state_estimator = self.data[state_estimator_index]['state_estimator']
            while state_estimator is None and state_estimator_index < len(self.timestamps) - 1:
                state_estimator_index += 1
                state_estimator = self.data[state_estimator_index]['state_estimator']
            if state_estimator is not None:
                if 'vWorld' in state_estimator:
                    vWorld_list.append(state_estimator['vWorld'])
                if 'vBody' in state_estimator:
                    vBody_list.append(state_estimator['vBody'])
                if 'rpy' in state_estimator:
                    rpy_list.append(state_estimator['rpy'])
                if 'omegaBody' in state_estimator:
                    omegaBody_list.append(state_estimator['omegaBody'])
                if 'omegaWorld' in state_estimator:
                    omegaWorld_list.append(state_estimator['omegaWorld'])
                if 'quat' in state_estimator:
                    quat_list.append(state_estimator['quat'])
                if 'aBody' in state_estimator:
                    aBody_list.append(state_estimator['aBody'])
                if 'aWorld' in state_estimator:
                    aWorld_list.append(state_estimator['aWorld'])

            # leg_control_data
            leg_control_data = self.data[leg_control_data_index]['leg_control_data']
            while leg_control_data is None and leg_control_data_index < len(self.timestamps) - 1:
                leg_control_data_index += 1
                leg_control_data = self.data[leg_control_data_index]['leg_control_data']
            if leg_control_data is not None:
                if 'q' in leg_control_data:
                    q_list.append(leg_control_data['q'])
                if 'qd' in leg_control_data:
                    qd_list.append(leg_control_data['qd'])
                if 'p' in leg_control_data:
                    p_list.append(leg_control_data['p'])
                if 'v' in leg_control_data:
                    v_list.append(leg_control_data['v'])
                if 'tau_est' in leg_control_data:
                    tau_est_list.append(leg_control_data['tau_est'])
                if 'force_est' in leg_control_data:
                    force_est_list.append(leg_control_data['force_est'])

            current_time += self.time_step

        # Check data lengths
        min_length = min(
            len(x_vel_des_list),
            len(y_vel_des_list),
            len(yaw_vel_des_list),
            len(rpy_des_list),
            len(vBody_list),
            len(aBody_list),
            len(omegaBody_list),
            len(rpy_list),
            len(q_list),
            len(qd_list),
            len(tau_est_list)
        )
        x_vel_des_list = x_vel_des_list[:min_length]
        y_vel_des_list = y_vel_des_list[:min_length]
        yaw_vel_des_list = yaw_vel_des_list[:min_length]
        rpy_des_list = rpy_des_list[:min_length]

        vBody_list = vBody_list[:min_length]
        aBody_list = aBody_list[:min_length]
        omegaBody_list = omegaBody_list[:min_length]
        rpy_list = rpy_list[:min_length]
        q_list = q_list[:min_length]
        qd_list = qd_list[:min_length]
        tau_est_list = tau_est_list[:min_length]

        # Merge data
        action = np.concatenate([
            np.array(x_vel_des_list).reshape(-1, 1),
            np.array(y_vel_des_list).reshape(-1, 1),
            np.array(yaw_vel_des_list).reshape(-1, 1),
            np.array(rpy_des_list)
        ], axis=-1)
        state = np.concatenate([
            np.array(vBody_list),
            np.array(aBody_list),
            np.array(omegaBody_list),
            np.array(rpy_list),
            np.array(q_list),
            np.array(qd_list),
            np.array(tau_est_list),
        ], axis=-1)
        x_and_u = np.concatenate([state, action], axis=-1)

        print(f"File from ## {self.file_name} ##, data shape ## {x_and_u.shape} ##")
        return x_and_u
