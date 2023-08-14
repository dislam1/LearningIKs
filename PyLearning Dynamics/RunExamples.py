def RunExample():
    scenarios = ['Run the main examples',
                'Run the Model Selection examples',
                'Run the M vs. L convergence test for the main examples']

    # Display the scenarios1
    for idx, scenario in enumerate(scenarios):
        print(f'[{idx+1}] {scenario}')

    print()


    # Prompt the user to pick a scenario
    scenario_idx = int(input('Pick a scenario to run: '))

    # Execute the chosen scenario
    if 1 <= scenario_idx <= len(scenarios):
        if scenario_idx == 1:
            from SOD_Utils.RunExamples_main import RunExamples_main
        elif scenario_idx == 2:
            RunExamples_modelSelection()
        elif scenario_idx == 3:
            RunExamples_MLTest()
    else:
        raise ValueError(f'Invalid user input (1 <= input <= {len(scenarios)})')
if __name__ == "__main__":
    RunExample()
    