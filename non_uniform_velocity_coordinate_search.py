import numpy as np
from tabulate import tabulate

# ----------------------------------------------------------------------------------------------------------------------

FIRST_SCENARIO_ITERATIONS = 100000                  # The number of iterations used to simulate the expected payoffs in the first step of the coordinate search.
NEXT_SCENARIO_ITERATIONS = 1000000                  # The number of iterations used to simulate the expected payoffs in the subsequent steps of the coordinate search.
FINAL_SCENARIO_ITERATIONS = 10000000                # The number of iterations used to simulate the expected payoffs in the final steps of the coordinate search.
FIRST_MAX_COORDINATE_SEARCH_ITERATIONS = 200000     # The maximum number of iterations for the first step of the coordinate search.
NEXT_MAX_COORDINATE_SEARCH_ITERATIONS = 2000000     # The maximum number of iterations for each subsequent step of the coordinate search.
MAX_EXPLOITABILITY = 0.00001                        # The program runs until it finds strategies that are exploitable for less than this. (this is for the final step)
DISCOUNT_RATE = 0.000005                            # Discount rate in the coordinate search.
SPECIFICITY = 0.2                                   # The specificity of the coordinate search.

SD_BY_VELOCITY = {              # Defining the possible velocity-choices and their corresponding SD.
    7: 1.5,
    8.5: 1.75,
    10: 2,
}
FRICTION = 0.1                  # Defining the friction factor.
SYMMETRIC = True                # Set this to True if all assumptions are symmetric across x=12, and false otherwise.

# Defining the aim options.
INITIAL_AIM_OPTIONS = []
x_values = [0.8, 2.4, 4.0, 5.6, 7.2, 8.8, 10.4, 12.0, 13.6, 15.2, 16.8, 18.4, 20.0, 21.6, 23.2]
y_values = [-0.8, 0.8, 2.4, 4.0, 5.6, 7.2]
velocity_values = list(SD_BY_VELOCITY.keys())
for x_value in x_values:
    for y_value in y_values:
        for velocity_value in velocity_values:
            INITIAL_AIM_OPTIONS.append(((x_value, y_value), velocity_value))

# Defining the area equations. (Key naming must follow this convention for the symmetry-functions to work)
AREA_EQUATIONS = {
    'commit_left': lambda x, y, v: ((x + 2 - 0.9*(7 if v <= 7 else v)) ** 2) / 3 + ((y + 0.5 - 0.25*(7 if v <= 7 else v)) ** 2) / 2 <= 6,
    'lean_left': lambda x, y, v: ((x - 7.5 - 0.25*v) ** 2) / 7 + ((y - 1.5 - 0.2*v) ** 2) / 7 <= 18 - 1.4*v,
    'stay_middle': lambda x, y, v: ((x - 12) ** 2) / 8 + ((y - 2 - 0.2*v) ** 2) / 8 <= 22.5 - 1.85*v,
    'lean_right': lambda x, y, v: ((x - 16.5 + 0.25*v) ** 2) / 7 + ((y - 1.5 - 0.2*v) ** 2) / 7 <= 18 - 1.4*v,
    'commit_right': lambda x, y, v: ((x - 26 + 0.9*(7 if v <= 7 else v)) ** 2) / 3 + ((y + 0.5 - 0.25*(7 if v <= 7 else v)) ** 2) / 2 <= 6,
}
AREA_OPTIONS = list(AREA_EQUATIONS.keys())

# Defining the boundaries for which hit-coordinates can result in a goal. (see appendix 1 for explanation)
LOWER_BOUND = 0.365
UPPER_BOUND = 7.751
LEFT_BOUND = 0.257
RIGHT_BOUND = 23.743

# ----------------------------------------------------------------------------------------------------------------------


def getStandardDeviation(aim_option):
    velocity = aim_option[1]
    sd = SD_BY_VELOCITY.get(velocity)
    return sd


def getHitCoordinate(aim_option):
    aim_coordinate = aim_option[0]
    sd = getStandardDeviation(aim_option)
    velocity = aim_option[1]
    hit_coordinate = np.random.normal(aim_coordinate, sd, size=2)
    hit_velocity = velocity
    if hit_coordinate[1] < LOWER_BOUND:
        hit_velocity = velocity - (LOWER_BOUND - hit_coordinate[1]) * FRICTION
        hit_velocity = max(hit_velocity, 0)
    hit_coordinate[1] = max(LOWER_BOUND, hit_coordinate[1])
    return hit_coordinate, hit_velocity


def isInsideScoringArea(hit_coordinate):
    return LEFT_BOUND < hit_coordinate[0] < RIGHT_BOUND and hit_coordinate[1] < UPPER_BOUND


def isSavedShot(hit_coordinate, hit_velocity, area):
    equation = AREA_EQUATIONS.get(area)
    return equation(hit_coordinate[0], hit_coordinate[1], hit_velocity)


def getScenarioEV(iterations, aim_option, area):
    goal = 0
    for _ in range(iterations):
        hit_coordinate, hit_velocity = getHitCoordinate(aim_option)
        if isInsideScoringArea(hit_coordinate) and not isSavedShot(hit_coordinate, hit_velocity, area):
            goal += 1
    return goal/iterations


def getScenarioData(iterations, aim_options, area_options, existing_data=None, replace=True):
    if existing_data is None:
        existing_data = {}
    all_scenarios = set((aim, area) for aim in aim_options for area in area_options)
    existing_scenarios = set(existing_data.keys())
    new_scenarios = all_scenarios - existing_scenarios
    scenario_data = existing_data.copy()
    scenarios_completed = 0
    print('Simulating expected values in each scenario.')
    for i in range(len(aim_options)):
        for j in range(len(area_options)):
            if replace:
                scenario_ev = getScenarioEV(iterations, aim_options[i], area_options[j])
                scenario_data[(aim_options[i], area_options[j])] = scenario_ev
                scenarios_completed += 1
                print('Completed scenario: ' + str(scenarios_completed) + '/' + str(len(aim_options)*len(area_options)))
            elif not replace and (aim_options[i], area_options[j]) in new_scenarios:
                scenario_ev = getScenarioEV(iterations, aim_options[i], area_options[j])
                scenario_data[(aim_options[i], area_options[j])] = scenario_ev
                scenarios_completed += 1
                print('Completed scenario: ' + str(scenarios_completed) + '/' + str(len(new_scenarios)))
    if SYMMETRIC:
        scenario_data = makeScenariosSymmetric(scenario_data, aim_options, area_options)
    return scenario_data


def makeScenariosSymmetric(scenario_data, aim_options, area_options):
    for i in range(len(aim_options)):
        for j in range(len(area_options)):
            x = aim_options[i][0][0]
            y = aim_options[i][0][1]
            velocity = aim_options[i][1]
            if x <= 12:
                symmetrical_x = 24 - x
                symmetrical_y = y
                if area_options[j] in ['commit_left', 'lean_left']:
                    symmetrical_area = area_options[j].replace('left', 'right')
                elif area_options[j] in ['commit_right', 'lean_right']:
                    symmetrical_area = area_options[j].replace('right', 'left')
                else:
                    symmetrical_area = area_options[j]
                symmetrical_EV = scenario_data[(((symmetrical_x, symmetrical_y), velocity), symmetrical_area)]
                EV = scenario_data[(((x, y), velocity), area_options[j])]
                avg_EV = (EV + symmetrical_EV) / 2
                scenario_data[(((x, y), velocity), area_options[j])] = avg_EV
                scenario_data[(((symmetrical_x, symmetrical_y), velocity), symmetrical_area)] = avg_EV
    return scenario_data


def getRegret(aim_coordinate, area, scenario_data, aim_options, area_options):
    num_options_pt = len(aim_options)
    num_options_gk = len(area_options)
    regret_pt = [0] * num_options_pt
    regret_gk = [0] * num_options_gk
    exp_value_pt = scenario_data[(aim_coordinate, area)]
    exp_value_gk = -exp_value_pt
    for i in range(num_options_pt):
        alt_exp_value_pt = scenario_data[(aim_options[i], area)]
        regret_pt[i] = alt_exp_value_pt - exp_value_pt
    for i in range(num_options_gk):
        alt_exp_value_gk = -scenario_data[(aim_coordinate, area_options[i])]
        regret_gk[i] = alt_exp_value_gk - exp_value_gk
    return regret_pt, regret_gk


def getStrategy(regret_sum, strategy_sum, num_options, discount_rate=0):
    strategy = [0] * num_options
    normalizing_sum = 0
    for i in range(num_options):
        if regret_sum[i] > 0:
            strategy[i] = regret_sum[i]
            normalizing_sum += strategy[i]
    for i in range(num_options):
        if normalizing_sum > 0:
            strategy[i] = strategy[i] / normalizing_sum
        else:
            strategy[i] = 1.0 / num_options
        strategy_sum[i] = strategy_sum[i] * (1-discount_rate) + strategy[i]
    return strategy, strategy_sum


def getAction(options, strategy):
    return options[np.random.choice(len(options), p=strategy)]


def makeStrategySymmetricPT(strategy, aim_options):
    symmetric_strategy = []
    for i in range(len(aim_options)):
        x = aim_options[i][0][0]
        y = aim_options[i][0][1]
        velocity = aim_options[i][1]
        if x == 12:
            symmetric_strategy.append(strategy[i])
        else:
            symmetrical_x = round(24 - x, 1)
            symmetrical_index = aim_options.index(((symmetrical_x, y), velocity))
            symmetrical_strategy = strategy[symmetrical_index]
            avg_strategy = (strategy[i] + symmetrical_strategy) / 2
            symmetric_strategy.append(avg_strategy)
    return symmetric_strategy


def makeStrategySymmetricGK(strategy, area_options):
    symmetrical_strategy = [0] * len(area_options)
    for i in range(len(area_options)):
        if area_options[i] in ['commit_left', 'lean_left']:
            symmetrical_area = area_options[i].replace('left', 'right')
        elif area_options[i] in ['commit_right', 'lean_right']:
            symmetrical_area = area_options[i].replace('right', 'left')
        else:
            symmetrical_area = area_options[i]
        symmetrical_index = area_options.index(symmetrical_area)
        if symmetrical_index != i:
            symmetrical_strategy[i] = (strategy[i] + strategy[symmetrical_index]) / 2
        else:
            symmetrical_strategy[i] = strategy[i]
    return symmetrical_strategy


def train(max_exploitability, scenario_data, aim_options, area_options, discount_rate=0, continue_indefinitely=False):
    print('Started training.')
    num_options_pt = len(aim_options)
    num_options_gk = len(area_options)
    strategy_pt = [1 / num_options_pt] * num_options_pt
    strategy_gk = [1 / num_options_gk] * num_options_gk
    regret_sum_pt = [0] * num_options_pt
    regret_sum_gk = [0] * num_options_gk
    strategy_sum_pt = [0] * num_options_pt
    strategy_sum_gk = [0] * num_options_gk
    optimal_strategy_pt = [0] * num_options_pt
    optimal_strategy_gk = [0] * num_options_gk

    exploitability_pt = 1
    exploitability_gk = 1
    i = 1

    while exploitability_pt > max_exploitability or exploitability_gk > max_exploitability or continue_indefinitely:
        aim = getAction(aim_options, strategy_pt)
        area = getAction(area_options, strategy_gk)

        for j in range(num_options_pt):
            regret_sum_pt[j] += getRegret(aim, area, scenario_data, aim_options, area_options)[0][j]

        for j in range(num_options_gk):
            regret_sum_gk[j] += getRegret(aim, area, scenario_data, aim_options, area_options)[1][j]

        strategy_pt, strategy_sum_pt = getStrategy(regret_sum_pt, strategy_sum_pt, num_options_pt, discount_rate)
        strategy_gk, strategy_sum_gk = getStrategy(regret_sum_gk, strategy_sum_gk, num_options_gk, discount_rate)

        if i % 10000 == 0:
            print(f'Completed {i:,} iterations.')

            normalizing_sum_pt = sum(strategy_sum_pt)
            normalizing_sum_gk = sum(strategy_sum_gk)

            for j in range(num_options_pt):
                optimal_strategy_pt[j] = strategy_sum_pt[j] / normalizing_sum_pt

            for j in range(num_options_gk):
                optimal_strategy_gk[j] = strategy_sum_gk[j] / normalizing_sum_gk

            if SYMMETRIC:
                optimal_strategy_pt = makeStrategySymmetricPT(optimal_strategy_pt, aim_options)
                optimal_strategy_gk = makeStrategySymmetricGK(optimal_strategy_gk, area_options)

            exp_value_pt, exp_value_gk, exploitability_pt, exploitability_gk = getExploitability(scenario_data, aim_options, area_options, optimal_strategy_pt, optimal_strategy_gk)

            if i % 50000 == 0 and (exploitability_pt > max_exploitability or exploitability_gk > max_exploitability or continue_indefinitely):
                showStrategies(aim_options, area_options, optimal_strategy_pt, optimal_strategy_gk, exp_value_pt, exp_value_gk, exploitability_pt, exploitability_gk)
        i += 1

    return optimal_strategy_pt, optimal_strategy_gk


def trainCoordinateSearch(scenario_data, aim_options, area_options, discount_rate=0):
    print('Started training.')
    num_options_pt = len(aim_options)
    num_options_gk = len(area_options)
    strategy_pt = [1 / num_options_pt] * num_options_pt
    strategy_gk = [1 / num_options_gk] * num_options_gk
    regret_sum_pt = [0] * num_options_pt
    regret_sum_gk = [0] * num_options_gk
    strategy_sum_pt = [0] * num_options_pt
    strategy_sum_gk = [0] * num_options_gk
    optimal_strategy_pt = [0] * num_options_pt
    optimal_strategy_gk = [0] * num_options_gk

    i = 1

    if firstIteration:
        max_iterations = FIRST_MAX_COORDINATE_SEARCH_ITERATIONS
    else:
        max_iterations = NEXT_MAX_COORDINATE_SEARCH_ITERATIONS

    while i <= max_iterations:
        aim = getAction(aim_options, strategy_pt)
        area = getAction(area_options, strategy_gk)

        for j in range(num_options_pt):
            regret_sum_pt[j] += getRegret(aim, area, scenario_data, aim_options, area_options)[0][j]

        for j in range(num_options_gk):
            regret_sum_gk[j] += getRegret(aim, area, scenario_data, aim_options, area_options)[1][j]

        strategy_pt, strategy_sum_pt = getStrategy(regret_sum_pt, strategy_sum_pt, num_options_pt, discount_rate)
        strategy_gk, strategy_sum_gk = getStrategy(regret_sum_gk, strategy_sum_gk, num_options_gk, discount_rate)

        if i % 10000 == 0:
            print(f'Completed {i:,} iterations.')

            normalizing_sum_pt = sum(strategy_sum_pt)
            normalizing_sum_gk = sum(strategy_sum_gk)

            for j in range(num_options_pt):
                optimal_strategy_pt[j] = strategy_sum_pt[j] / normalizing_sum_pt

            for j in range(num_options_gk):
                optimal_strategy_gk[j] = strategy_sum_gk[j] / normalizing_sum_gk

            if SYMMETRIC:
                optimal_strategy_pt = makeStrategySymmetricPT(optimal_strategy_pt, aim_options)
                optimal_strategy_gk = makeStrategySymmetricGK(optimal_strategy_gk, area_options)

            low_freq_options = 0
            for j in range(num_options_pt):
                if 0.01 > optimal_strategy_pt[j] >= 0.0005:
                    low_freq_options += 1

            if low_freq_options == 0:
                break

            if i % 50000 == 0 and i > 0:
                exp_value_pt, exp_value_gk, exploitability_pt, exploitability_gk = getExploitability(scenario_data, aim_options, area_options, optimal_strategy_pt, optimal_strategy_gk)
                showStrategies(aim_options, area_options, optimal_strategy_pt, optimal_strategy_gk, exp_value_pt, exp_value_gk, exploitability_pt, exploitability_gk)

        i += 1

    return optimal_strategy_pt, optimal_strategy_gk


def getStrategyEVs(scenario_data, aim_options, area_options, strategy_pt, strategy_gk):
    exp_value_pt = 0
    for i in range(len(aim_options)):
        for j in range(len(area_options)):
            exp_value_pt += \
                scenario_data[(aim_options[i], area_options[j])] * strategy_pt[i] * strategy_gk[j]
    exp_value_gk = -exp_value_pt
    return exp_value_pt, exp_value_gk


def getExploitability(scenario_data, aim_options, area_options, strategy_pt, strategy_gk):
    num_options_pt = len(aim_options)
    num_options_gk = len(area_options)
    exp_value_pt, exp_value_gk = \
        getStrategyEVs(scenario_data, aim_options, area_options, strategy_pt, strategy_gk)
    best_response_ev_pt = 0
    best_response_ev_gk = -1

    for i in range(num_options_pt):
        pure_strategy_pt = [0] * num_options_pt
        pure_strategy_pt[i] = 1
        option_ev = getStrategyEVs(scenario_data,
                                   aim_options,
                                   area_options,
                                   pure_strategy_pt,
                                   strategy_gk)[0]
        if option_ev > best_response_ev_pt:
            best_response_ev_pt = option_ev

    for i in range(num_options_gk):
        pure_strategy_gk = [0] * num_options_gk
        pure_strategy_gk[i] = 1
        option_ev = getStrategyEVs(scenario_data,
                                   aim_options,
                                   area_options,
                                   strategy_pt,
                                   pure_strategy_gk)[1]
        if option_ev > best_response_ev_gk:
            best_response_ev_gk = option_ev

    exploitability_pt = best_response_ev_gk - exp_value_gk
    exploitability_gk = best_response_ev_pt - exp_value_pt

    return exp_value_pt, exp_value_gk, exploitability_pt, exploitability_gk


def showStrategies(aim_options, area_options, strategy_pt, strategy_gk, exp_value_pt, exp_value_gk, exploitability_pt, exploitability_gk):
    num_options_pt = len(aim_options)
    num_options_gk = len(area_options)
    pt_table = []
    gk_table = []
    ev_table = [['Penalty taker', round(exp_value_pt, 5), round(exploitability_pt, 5)],
                ['Goalkeeper', round(exp_value_gk, 5), round(exploitability_gk, 5)]]

    for i in range(0, num_options_pt):
        pt_table.append([aim_options[i], round(strategy_pt[i], 3)])
    for i in range(0, num_options_gk):
        gk_table.append([area_options[i], round(strategy_gk[i], 3)])

    print('**************************')
    print('PENALTY TAKER STRATEGY\n' + tabulate(pt_table, headers=['Aim point', 'Probability']))
    print('**************************')
    print('GOALKEEPER STRATEGY\n' + tabulate(gk_table, headers=['Area', 'Probability']))
    print('**************************************************')
    print(tabulate(ev_table, headers=['Player', 'Expected Value', 'Exploitability']))
    print('**************************************************')


def getOtherViableOptions(scenario_data, aim_options, area_options, strategy_gk, exp_value_pt):
    other_viable_options = []
    for (aim_option, area_option) in scenario_data.keys():
        pure_strategy_pt = [0] * len(aim_options)
        pure_strategy_pt[aim_options.index(aim_option)] = 1
        exp_value_option, _ = getStrategyEVs(scenario_data, aim_options, area_options, pure_strategy_pt, strategy_gk)
        if exp_value_option >= exp_value_pt:
            other_viable_options.append(aim_option)
    return other_viable_options


def getDistanceToNearestPoint(aim_option, aim_options):
    nearest_distance = 1.6
    distance = 1.6
    point = aim_option[0]
    for option in aim_options:
        for i in range(2):
            if option[0][i] != point[i]:
                distance = round(abs(option[0][i] - point[i]), 1)
            if nearest_distance > distance:
                nearest_distance = distance
    return nearest_distance


def getNewAimOptions(aim_option, distance, aim_options):
    point = aim_option[0]
    velocity = aim_option[1]
    values = [-distance, 0, distance]
    is_surrounded = False
    surrounding_points = [((round(point[0] - i, 1), round(point[1] - j, 1)), velocity) for i in values for j in values]
    if set(surrounding_points).issubset(set(aim_options)):
        is_surrounded = True
        if distance > SPECIFICITY:
            values = [-distance/2, 0, distance/2]
    new_aim_options = [((round(point[0] - i, 1), round(point[1] - j, 1)), velocity) for i in values for j in values]
    new_aim_options = list(set(new_aim_options))
    new_aim_options.sort()
    return new_aim_options, is_surrounded


def getFrequentAimOptions(strategy, aim_options):
    frequent_aim_options = []
    for i in range(len(strategy)):
        if strategy[i] >= 0.0005:
            frequent_aim_options.append(aim_options[i])
    return frequent_aim_options


allAimOptions = INITIAL_AIM_OPTIONS
allNewAimOptions = INITIAL_AIM_OPTIONS
firstIteration = True
scenarioData = getScenarioData(FIRST_SCENARIO_ITERATIONS, INITIAL_AIM_OPTIONS, AREA_OPTIONS)

while True:
    optimalStrategyPT, optimalStrategyGK = trainCoordinateSearch(scenarioData, allNewAimOptions, AREA_OPTIONS, DISCOUNT_RATE)
    expValuePT, expValueGK, exploitabilityPT, exploitabilityGK = getExploitability(scenarioData, allNewAimOptions, AREA_OPTIONS, optimalStrategyPT, optimalStrategyGK)
    showStrategies(allNewAimOptions, AREA_OPTIONS, optimalStrategyPT, optimalStrategyGK, expValuePT, expValueGK, exploitabilityPT, exploitabilityGK)

    frequentAimOptions = getFrequentAimOptions(optimalStrategyPT, allNewAimOptions)
    otherViableOptions = getOtherViableOptions(scenarioData, allAimOptions, AREA_OPTIONS, optimalStrategyGK, expValuePT)
    viableOptions = list(set(frequentAimOptions + otherViableOptions))
    viableOptions.sort()

    allNewAimOptions = []
    num_converged = 0
    for option in viableOptions:
        distance = getDistanceToNearestPoint(option, allAimOptions)
        newAimOptions, isSurrounded = getNewAimOptions(option, distance, allAimOptions)
        allNewAimOptions.extend(newAimOptions)

        if (option in frequentAimOptions) and isSurrounded and distance == SPECIFICITY:
            num_converged += 1

    allNewAimOptions = list(set(allNewAimOptions))
    allNewAimOptions.sort()

    allAimOptions.extend(allNewAimOptions)
    allAimOptions = list(set(allAimOptions))
    allAimOptions.sort()

    if len(viableOptions) == len(frequentAimOptions) == num_converged:
        print('Completed the option search.')
        break

    if firstIteration:
        scenarioData = getScenarioData(NEXT_SCENARIO_ITERATIONS, allNewAimOptions, AREA_OPTIONS, existing_data=scenarioData, replace=True)
        firstIteration = False
    else:
        scenarioData = getScenarioData(NEXT_SCENARIO_ITERATIONS, allNewAimOptions, AREA_OPTIONS, existing_data=scenarioData, replace=False)


scenarioData = getScenarioData(FINAL_SCENARIO_ITERATIONS, frequentAimOptions, AREA_OPTIONS, existing_data=scenarioData, replace=True)
optimalStrategyPT, optimalStrategyGK = train(MAX_EXPLOITABILITY, scenarioData, frequentAimOptions, AREA_OPTIONS)
expValuePT, expValueGK, exploitabilityPT, exploitabilityGK = getExploitability(scenarioData, frequentAimOptions, AREA_OPTIONS, optimalStrategyPT, optimalStrategyGK)
showStrategies(frequentAimOptions, AREA_OPTIONS, optimalStrategyPT, optimalStrategyGK, expValuePT, expValueGK, exploitabilityPT, exploitabilityGK)
