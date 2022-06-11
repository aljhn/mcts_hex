def parse_config(config_file_path):
    config = {}

    f = open(config_file_path)
    lines = f.readlines()

    for line in lines:
        line = line.lower() # To lowercase
        line = line[:line.find("#")] # Remove comments
        line = "".join(line.split()) # Remove all whitespace

        if len(line) == 0: # Nothing here
            continue
        
        line = line.split(":")

        if line[1][0] == "[":
            config[line[0]] = []
            values = line[1][1:-1].split(",")
            for value in values:
                config[line[0]].append(value)
        else:
            config[line[0]] = line[1]

    f.close()

    return config
