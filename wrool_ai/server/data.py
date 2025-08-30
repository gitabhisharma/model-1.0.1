import secrets
print(secrets.token_hex(32))


import pandas as pd

data={
    "User_name":[('')],
    "User_password":[('')]
}

df = pd.read_json('data.json')

print(df)
