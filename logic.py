import os

from Agent import Agent

""" logic enthält allgemeinere Methoden, die man über mehrere Klassen verwenden kann """

""" gedacht um aus DQN_attributes hidden size abzulesen, damit Agent mit korrekter Netzgröße geladen werden kann """
def get_attributes_from_file(file) -> dict:
    """ Liest integer Werte aus einer txt-Datei ab """
    attributes : dict = {}

    with open(file, "r") as f:
        for line in f:
            if "=" in line:
                key, value = line.split("=")
                attributes[key.strip()] = value.strip()

    for attribute in attributes.keys():
        attributes[attribute] : int = int(attributes[attribute])

    return attributes

def select_path(path, prefix = None) -> str:
    """ Lässt Nutzer einen Pfad aus einer Auflistung an Pfaden in einem Ordner wählen """
    path : str = f"{path}/"

    if prefix:
        directory : list[str] = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)) and d.startswith(prefix)]
        directory.sort(key=lambda d: int(d.split('_')[-1]) if d.split('_')[-1].isdigit() else 0)
    else:
        directory : list[str] = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        directory.sort()

    directory_dict : dict = {}

    print("Einen der folgenden Optionen auswählen: ")

    for i in range(len(directory)):
        print(f"{i+1}. {directory[i]}")
        directory_dict[i] = directory[i]

    while True:
        user_input : str = input("> ")

        try:
            user_input = directory_dict[int(user_input) - 1]
        except ValueError:
            user_input = str(user_input)

        if user_input in directory_dict.values() or user_input in directory_dict.keys():
            print(f"{user_input} ausgewählt")
            return f"{path}/{user_input}"
        else:
            print("Gewählte Option nicht in Liste")

def select_agent() -> Agent:
    """ Lässt Nutzer Modell wählen und lädt einen entsprechenden Agenten """
    selected_trainer : str = select_path("training", "trainer_")
    selected_model : str = select_path(selected_trainer)

    agents : list[str] = [d for d in os.listdir(selected_model) if d.startswith("final_agent")]
    selected_agent : str = f"{selected_model}/{sorted(agents)[0]}"

    dqn_attributes : dict = get_attributes_from_file(f"{selected_trainer}/DQN_attributes")

    agent : Agent = Agent(hidden_size = dqn_attributes["hidden size"])
    agent.load_model(f"{selected_agent}")

    return agent

def next_directory(directory, prefix) -> str:
    """ Generiert für den gewählten Prefix die nächste freie Bezeichnung nach dem Schema: name, name_1, name_2, ... """
    i : int = 0

    if os.path.exists(f"{directory}/{prefix}"):
        i += 1

    while os.path.exists(f"{directory}/{prefix}_{i}"):
        i += 1

    if i == 0:
        new_dir : str = prefix
        return new_dir
    else:
        new_dir : str = f"{prefix}_{i}"
        return new_dir

def select_value(prompt, default = None, value_type = None) -> str | int | float:
    """ Lässt Benutzer einen Wert eines gewünschten Datentyps eingeben """
    while True:
        if default is not None:
            user_input : str = input(f"{prompt} (Standardwert: {default}): ")
        else:
            user_input : str = input(prompt)

        if not user_input:
            return default
        elif value_type == "int":
            try:
                user_input : int = int(user_input)
            except ValueError:
                print("Bitte Integer eingeben.")
        elif value_type == "float":
            try:
                user_input : float = float(user_input)
            except ValueError:
                print("Bitte Float eingeben.")

        return user_input
