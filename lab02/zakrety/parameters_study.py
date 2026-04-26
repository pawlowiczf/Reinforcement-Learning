import matplotlib.pyplot as plt
import numpy as np
import itertools

from problem import Action, available_actions, Corner, Driver, Experiment, Environment, State
from solution import OffPolicyNStepSarsaDriver

def parametric_study():
    alphas = [0.3, 0.5, 0.7, 0.9]
    step_nos = [2, 4]
    results = {}

    for n, alpha in itertools.product(step_nos, alphas):
        print(f"n={n}, alpha={alpha}...")
        driver = OffPolicyNStepSarsaDriver(
            step_no=n,
            step_size=alpha,
            experiment_rate=0.1,
            discount_factor=1.0,
        )
        experiment = Experiment(
            environment=Environment(
                corner=Corner(name='corner_c'),
                steering_fail_chance=0.01,
            ),
            driver=driver,
            number_of_episodes=1000,
        )
        experiment.run()

        driver.evaluation_mode = True
        eval_experiment = Experiment(
            environment=Environment(
                corner=Corner(name='corner_c'),
                steering_fail_chance=0.01,
            ),
            driver=driver,
            number_of_episodes=100,
        )
        eval_experiment.run()

        results[(n, alpha)] = np.mean(eval_experiment.penalties)
    #

    for n in step_nos:
        y = [results[(n, alpha)] for alpha in alphas]
        plt.plot(alphas, y, label=f'n={n}', marker='o')

    plt.xlabel('α (step_size)')
    plt.legend()
    plt.savefig('plots_parameters/parametric_study.png', dpi=300)
    plt.clf()

parametric_study()