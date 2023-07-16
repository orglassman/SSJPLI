import csv
import sqlite3
from random import choices


def get_cursor():
    connection = sqlite3.connect(':memory:')
    cursor = connection.cursor()
    return cursor


def create_table(cursor, table_name, columns):
    column_defs = ', '.join([f'{col} TEXT' for col in columns])
    create_table_query = f'CREATE TABLE {table_name} ({column_defs})'
    cursor.execute(create_table_query)


def insert_data(cursor, table_name, columns, data):
    placeholders = ', '.join(['?' for _ in columns])
    insert_query = f'INSERT INTO {table_name} VALUES ({placeholders})'
    cursor.executemany(insert_query, data)


def join(cursor, table_names, column_names, on):
    """inner join on arbitrary number of tables"""
    select_statement = ', '.join([f'{table}.{column}' for table, column in zip(table_names, column_names)])
    select_statement += f', {table_names[0]}.{on}'

    join_condition = ' AND '.join([f'{table_names[i]}.{on} = {table_names[i + 1]}.{on}' for i in range(len(table_names) - 1)])

    query = f'''SELECT {select_statement} \
                FROM {table_names[0]} \
   {' '.join([f'INNER JOIN {table} ON {table_names[i]}.{on} = {table}.{on}' for i, table in enumerate(table_names[1:])])} \
                WHERE {join_condition} GROUP BY {select_statement}'''

    cursor.execute(query)

    # legacy pairwise
    # cursor.execute(f'''SELECT {name1}.{cols1}, {name2}.{cols2}, GROUP_CONCAT({name1}.TID, ',')
    #     FROM {name1} INNER JOIN {name2} ON {name1}.{on} = {name2}.{on}
    #     GROUP BY {name1}.{cols1}, {name2}.{cols2}'''

    rows = cursor.fetchall()
    return rows


def sample_data(cursor, table_name, sample_size, sampling_weights):
    """
    weighted sample - thus we can incorporate SSJ
    """
    query = f'SELECT * FROM {table_name} ORDER BY RANDOM()'
    cursor.execute(query)
    all_rows = cursor.fetchall()
    sampled_data = choices(all_rows, weights=sampling_weights, k=sample_size)
    return sampled_data


def get_distribution_from_join(cursor, X):
    # prepare query
    column_names = list(X)
    table_names = [f'T_{x}' for x in column_names]

    rows = join(cursor, table_names, column_names, on='TID')

    # build distribution
    tmp = {}
    for row in rows:
        x = ','.join(row[:-1])
        index = row[-1]
        if x in tmp.keys():
            tmp[x].append(index)
        else:
            tmp[x] = [index]

    N = len(rows)
    P = {k: len(v) / N for k, v in tmp.items()}
    return P


if __name__ == '__main__':
    csv_file = 'C:\\Users\\orgla\\Desktop\\Study\\J_Divergence_ST_formulation\\Datasets\\Cinema\\cinema.csv'
    name = 'cinema'
    # connect to in-memory SQLite db
    connection = sqlite3.connect(':memory:')
    cursor = connection.cursor()

    # set up db table
    table_name = 'connect4'
    N = 0
    with open(csv_file, 'r') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)  # read header
        create_table(cursor, table_name, header)

        # load data onto db
        data = [row for row in csv_reader]
        N = len(data)
        insert_data(cursor, table_name, header, data)

    # commit changes
    connection.commit()

    sample_size = 10
    weights = [1] * N
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]

    sampled_data = sample_data(cursor, table_name, sample_size, normalized_weights)
    for row in sampled_data:
        print(row)

    # close connection
    connection.close()
