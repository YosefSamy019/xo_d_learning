from quantized_q_table import *
import numpy as np

if __name__ == "__main__":
    smax = np.array([100, 3.14, 1])
    smin = np.array([0, -3.14, -1])

    q_table = QuantizedQTable(
        states_max_vals=smax,
        states_min_vals=smin,
        n_actions=4,
        n_quantization_lvls=[101, 91, 21],
        cache_folder="cache_test",
        state_names=['angle', 'height', 'velocity'],
        action_names=['up', 'down', 'left', 'right'],
        use_access_counter=True
    )

    for i in range(5_000):
        q_table.set_val(
            np.random.rand(3) * (smax - smin) + smin,
            np.random.randint(4),
            np.random.rand() * 10,
        )
    for i in range(5_000):
        q_table.get_val(
            np.random.rand(3) * (smax - smin) + smin,
            np.random.randint(4),
        )

    q_table.save_to_cache()
    print(q_table.get_state_pins())

    # print(q_table.export_2_df())
