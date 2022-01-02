import datetime as datetime
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import locale

locale.setlocale(locale.LC_ALL, 'de_DE.UTF-8')
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# load csv
df = pd.read_csv('races.csv', sep=';')


def json_to_prob(json: str, weather: str):
    """
    convert forecast json into float
    :param json: json string
    :param weather: weather type, e.g. 'sunny'
    :return: float belonging to weather type
    """
    return float(json.split(f'{weather}";i:')[1].split(';')[0]) / 100.0


# convert forecast into four new columns
df['sunny'] = df['forecast'].map(lambda x: json_to_prob(x, weather='sunny'))
df['rainy'] = df['forecast'].map(lambda x: json_to_prob(x, weather='rainy'))
df['thundery'] = df['forecast'].map(lambda x: json_to_prob(x, weather='thundery'))
df['snowy'] = df['forecast'].map(lambda x: json_to_prob(x, weather='snowy'))
df.drop('forecast', axis=1, inplace=True)


def fuel_to_float(x: str):
    """
    0.234 -> 0.234
    Jan 19 -> 1.19
    14. Feb -> 14.2
    :param x:
    :return: float
    """
    if '. ' in x:
        a, b = x.split('. ')
    elif ' ' in x:
        b, a = x.split(' ')
    else:
        return float(x)
    b = 'Mär' if b == 'Mrz' else b
    b = datetime.datetime.strptime(b, "%b").month
    return b + float(a) / 100.0  # this assumes Jan 89 = 01. 89 instead of e.g. 01. 1989



# convert fuel consumption into floats
df['fuel_consumption'] = df['fuel_consumption'].map(fuel_to_float)

# convert race_driven to seconds since 01. Jan 2012
df['race_driven'] = df['race_driven'].map(lambda x: None if x == '0000-00-00 00:00:00' else
                                          (datetime.datetime.strptime(x, '%d.%m.%Y %H:%M') -
                                           datetime.datetime(2012, 1, 1)).total_seconds())

# since the data seems to be sorted, we can interpolate missing values
df['race_driven'].interpolate(method='linear', inplace=True)

# convert back to datetime
df['race_driven'] = df['race_driven'].map(lambda x: datetime.datetime(2012, 1, 1) + datetime.timedelta(0, x))

print(df.head(10))

# df = df.loc[df['status'] == 'finished']
# print('Mean')
# print(df.groupby('track_id').mean())
# print('Median')
# print(df.groupby('track_id')['money'].quantile(q=0.8))
# print('Min')
# print(df.groupby('track_id').min())
# print('Max')
# print(df.groupby('track_id').max())

df.to_csv('races_cleaned.csv', sep=';', index=False)
