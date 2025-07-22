import pandas as pd
import re
from datetime import datetime




def get_date_from_period(periodo):
    """
    Convierte un periodo tipo '2019A', '2020B' o '2021' en una fecha estimada.

    Reglas:
    - 'A' → primer semestre → junio
    - 'B' → segundo semestre → diciembre
    - sin letra → se asume enero del año completo

    Retorna string con formato 'YYYY-MM-DD' o None si no se puede interpretar.
    """
    if pd.isna(periodo):
        return None

    periodo = str(periodo).strip().upper()

    match = re.match(r'^(\d{4})([AB])$', periodo)
    if match:
        año, semestre = match.groups()
        mes = '06' if semestre == 'A' else '12'
        return f"{año}-{mes}-01"

    match = re.match(r'^(\d{4})$', periodo)
    if match:
        return f"{periodo}-01-01"

    return None