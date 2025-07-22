import pandas as pd

# Cargar el archivo CSV
df  = pd.read_csv('./data/paises.csv')

print(df)

df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
# Nombre de la tabla
table_name = "country"

# Generar los scripts de inserción
insert_statements = []
for _, row in df.iterrows():
    insert = f"INSERT INTO {table_name} (nombre, name, iso2, iso3, phone_code) VALUES (" \
             f"'{row['nombre']}', '{row['name']}', '{row['iso2']}', '{row['iso3']}', '{row['phone_code']}');"
    insert_statements.append(insert)
















    

# Guardar los scripts en un archivo .sql
with open("insert_paises.sql", "w", encoding="utf-8") as f:
    f.write("\n".join(insert_statements))

print("Script generado con éxito.")