def SelectExample(Params, Examples):
    while True:
        if Params is None or 'ExampleName' not in Params:
            print('\nExamples:')
            for k, example in enumerate(Examples):
                sysinfo = example["sys_info"]
                print(f' [{k + 1}] {sysinfo.get("name")}')
            print('\n')

            try:
                ExampleIdx = int(input('Pick an example to run: '))
                ExampleIdx -= 1  # Adjust for 0-based indexing in Python
                if 0 <= ExampleIdx < len(Examples):
                    print(f'\nRunning {Examples[ExampleIdx]["sys_info"]["name"]}\n')
                    return ExampleIdx
                else:
                    print('Invalid example index. Please try again.')
            except ValueError:
                print('Invalid input. Please enter a valid integer.')
        else:
            example_names = [ex["sys_info"]["name"] for ex in Examples]
            try:
                ExampleIdx = example_names.index(Params['ExampleName'])
                print(f'\nRunning {Params["ExampleName"]}\n')
                return ExampleIdx
            except ValueError:
                raise ValueError(f'ExampleName in Params has no matching entry in all the pre-set examples!!')

    # Return a default value if no valid example is found
    return None
