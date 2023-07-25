import random
import sqlite3
import traceback

squ_params = {}

def set_params(**kwargs):
    """this has to be done prior to SSJing"""
    for k, v in kwargs.items():
        squ_params[k] = v


def print_info(cursor):
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    print(cursor.fetchall())


def get_cursor():
    connection = sqlite3.connect(':memory:')
    cursor = connection.cursor()
    return cursor


def create_table(cursor, table_name, columns):
    column_defs = ', '.join([f'"{col}" TEXT' for col in columns])
    create_table_query = f'CREATE TABLE "{table_name}" ({column_defs})'
    try:
        cursor.execute(create_table_query)
    except Exception:
        print(traceback.format_exc())


def insert_data(cursor, table_name, columns, data):
    placeholders = ', '.join(['?' for _ in columns])
    insert_query = f'INSERT INTO "{table_name}" VALUES ({placeholders})'
    cursor.executemany(insert_query, data)


def equijoin(cursor, table_names, column_names, on):
    """inner equijoin on arbitrary number of tables"""
    select_statement = ', '.join([f'{table}.{column}' for table, column in zip(table_names, column_names)])
    select_statement += f', {table_names[0]}.{on}'

    join_condition = ' AND '.join([f'{table_names[i]}.{on} = {table_names[i + 1]}.{on}' for i in range(len(table_names) - 1)])

    query = f'''SELECT {select_statement} \
                FROM {table_names[0]} \
   {' '.join([f'INNER JOIN {table} ON {table_names[i]}.{on} = {table}.{on}' for i, table in enumerate(table_names[1:])])} \
                WHERE {join_condition} GROUP BY {select_statement}'''

    cursor.execute(query)
    rows = cursor.fetchall()
    return rows


def PLI_join_cnts(cursor, table_names, X):
    output_name = f'CNT_{",".join(X)}'
    new_colname = f'{",".join([x.lower() for x in X])}'
    concat_args = [f'"{alias}"."{alias.lower()}"' for alias in X]
    concat_statement = ' || "," || '.join(concat_args)


    aliased_tables = ', '.join([f'"{table}" "{alias}"' for table, alias in zip(table_names, X)])


    join_condition = ' AND '.join([f'"{X[i]}".tid = "{X[i + 1]}".tid' for i in range(len(X) - 1)])


    query = f'''CREATE TABLE "{output_name}" AS \
                       SELECT {concat_statement} as "{new_colname}", count(*) as cnt \
                       FROM {aliased_tables} \
                       WHERE {join_condition} \
                       GROUP BY "{new_colname}" HAVING count(*) > 1'''


    cursor.execute(query)
    cursor.execute(f'SELECT "{new_colname}", cnt FROM "{output_name}"')
    return cursor.fetchall()


def PLI_join_tids(cursor, table_names, X):
    """
    Z - previously joined CNT table
    Select Hash(A.val, B.val) as val,A.tid as tid
    From TIDα A, TIDβ B, CNTα∪β Z
    Where A.tid = B.tid and Hash(A.val, B.val) = Z.va
    """
    merged_X = ",".join(X)
    output_name = f'TID_{merged_X}'
    Z = f'CNT_{merged_X}'

    concat_args = [f'"{col}"."{col.lower()}"' for col in X]
    concat_statement = ' || "," || '.join(concat_args)

    aliased_tables = ', '.join([f'"{table}" "{alias}"' for table, alias in zip(table_names, X)])

    join_condition = ' AND '.join([f'{X[i]}.tid = {X[i + 1]}.tid' for i in range(len(X) - 1)])
    join_condition += f' AND {X.lower()} = {Z}.{X.lower()}'

    query = f'''CREATE TABLE {output_name} AS \
                        SELECT {concat_statement} as {X.lower()}, {X[0]}.tid as tid \
                        FROM {aliased_tables},{Z} \
                        WHERE {join_condition} '''

    cursor.execute(query)


def SSJ_build_pairwise_tables(cursor, lhs_weights, rhs_weights, lhs_tid_name, rhs_tid_name):

    # init new TID and CNT in-memory
    new_tid_name, new_cnt_name = SSJ_init_new_tables(cursor, lhs_tid_name, rhs_tid_name)
    tid_data_to_insert = []
    cnt_data_to_insert = []

    # for querying T_A, T_B
    colname_A = lhs_tid_name.split('_')[-1].lower()
    colname_B = rhs_tid_name.split('_')[-1].lower()


    # flow control
    seen = {}
    num_sampled_entries = 0
    res = {}

    # main loop
    target_N = int(squ_params['coverage'] * squ_params['N'])
    while num_sampled_entries < target_N:
        # sample, discard if seen
        a = random.choices(list(lhs_weights.keys()), weights=lhs_weights.values())[0]
        b = random.choices(list(rhs_weights.keys()), weights=rhs_weights.values())[0]
        x = f'{a},{b}'
        if x in seen.keys():
            continue
        seen[x] = 1

        # query tables
        cursor.execute(
                        f'SELECT A.tid \
                        FROM "{lhs_tid_name}" A, "{rhs_tid_name}" B \
                        WHERE A.tid = B.tid AND A."{colname_A}" = "{a}" AND B."{colname_B}" = "{b}"'
        )
        tids = [r[0] for r in cursor.fetchall()]
        Nab = len(tids)
        if not tids:
            continue

        # update tables
        tid_data_to_insert += [(x, t) for t in tids]
        cnt_data_to_insert.append((x, Nab))

        # flow control
        num_sampled_entries += Nab
        res[x] = Nab

        # update sampling distributions
        lhs_weights[a] -= Nab
        if lhs_weights[a] <= 0:
            del lhs_weights[a]

        rhs_weights[b] -= Nab
        if rhs_weights[b] <= 0:
            del rhs_weights[b]

        if (not lhs_weights.keys()) or (not rhs_weights.keys()):
            break

    # insert data to new tid
    tid_colnames = [f'{colname_A},{colname_B}', 'tid']
    cnt_colnames = [f'{colname_A},{colname_B}', 'cnt']
    insert_data(cursor, new_tid_name, tid_colnames, tid_data_to_insert)
    insert_data(cursor, new_tid_name, cnt_colnames, cnt_data_to_insert)

    return res, new_tid_name, num_sampled_entries


def SSJ_get_weights(cursor, cnt_name):
    instance_colname = cnt_name.split('_')[-1].lower()
    query = f'SELECT "{instance_colname}", cnt FROM "{cnt_name}"'
    cursor.execute(query)
    rows = cursor.fetchall()
    weights = {r[0]: int(r[1]) for r in rows}
    return weights


def SSJ_init_new_tables(cursor, lhs_tid_name, rhs_tid_name):
    A = lhs_tid_name.split('_')[-1]
    B = rhs_tid_name.split('_')[-1]
    colname_A = A.lower()
    colname_B = B.lower()
    new_tablename = f'{A},{B}'
    new_colname = f'{colname_A},{colname_B}'
    new_tid_name = f'TID_{new_tablename}'
    new_cnt_name = f'CNT_{new_tablename}'
    create_table(cursor, new_tid_name, [new_colname, 'tid'])
    create_table(cursor, new_cnt_name, [new_colname, 'cnt'])
    return new_tid_name, new_cnt_name


def hash_function(*args):
    """ example hash function using Python built-in hash() """
    return hash(' '.join(str(arg) for arg in args))