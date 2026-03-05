import os

from Agent import Agent

def get_attributes_from_file(file):
    attributes = {}

    with open(file, "r") as f:
        for line in f:
            if "=" in line:
                key, value = line.split("=")
                attributes[key.strip()] = value.strip()

    for attribute in attributes.keys():
        attributes[attribute] = int(attributes[attribute])

    return attributes

def select_path(path, prefix = None):
    path = f"{path}/"

    if prefix:
        directory = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)) and d.startswith(prefix)]
        directory.sort(key=lambda d: int(d.split('_')[-1]) if d.split('_')[-1].isdigit() else 0)
    else:
        directory = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        directory.sort()
    directory_dict = {}

    print("Einen der folgenden Optionen auswählen: ")
    for i in range(len(directory)):
        print(f"{i+1}. {directory[i]}")
        directory_dict[i] = directory[i]

    while True:
        user_input = input(">")

        try:
            user_input = directory_dict[int(user_input) - 1]
        except ValueError:
            user_input = str(user_input)

        if user_input in directory_dict.values() or user_input in directory_dict.keys():
            print(f"{user_input} ausgewählt")
            return f"{path}/{user_input}"
        else:
            print("Gewählte Option nicht in Liste")

def select_agent():
    selected_trainer = select_path("training", "trainer_")
    selected_model = select_path(selected_trainer)

    agents = [d for d in os.listdir(selected_model) if d.startswith("final_agent")]
    selected_agent = f"{selected_model}/{sorted(agents)[0]}"

    dqn_attributes = get_attributes_from_file(f"{selected_trainer}/DQN_attributes")

    agent = Agent(hidden_size = dqn_attributes["hidden size"])
    agent.load_model(f"{selected_agent}")

    return agent

def next_directory(directory, prefix):
    i = 0
    if os.path.exists(f"{directory}/{prefix}"):
        i += 1

    while os.path.exists(f"{directory}/{prefix}_{i}"):
        i += 1

    if i == 0:
        new_dir = prefix
        return new_dir
    else:
        new_dir = f"{prefix}_{i}"
        return new_dir

def select_value(prompt, default = None, value_type = None):
    while True:
        if default is not None:
            user_input = input(f"{prompt} (Standardwert: {default}): ")
        else:
            user_input = input(prompt)

        if value_type == "int":
            try:
                user_input = int(user_input)
            except ValueError:
                print("Bitte Integer eingeben.")
        elif value_type == "float":
            try:
                user_input = float(user_input)
            except ValueError:
                print("Bitte Float eingeben.")

        return user_input

if __name__ == "__main__":
    print(next_directory("training/trainer_16", "base_model"))
