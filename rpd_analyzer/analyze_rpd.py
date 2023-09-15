import sqlite3
import argparse
import pandas as pd
from collections import defaultdict
import csv

def extract_additional_metrics(df_table, rpd_file):

    """_summary_
        df_table: Pandas DF
        rpd_file: str
    """

    kernel_launch_info = defaultdict(int)
    kernel_name_set = set()
    col_to_analyze = ['kernelName', 'gridX', 'gridY', 'gridZ', 
                      'workgroupX', 'workgroupY', 'workgroupZ'] # instantiated here for visualization
    
    for i in range(len(df_table)):

        kernel_name = str(df_table.loc[i, 'kernelName'])
        grid_x = str(df_table.loc[i, 'gridX'])
        grid_y = str(df_table.loc[i, 'gridY'])
        grid_z = str(df_table.loc[i, 'gridZ'])
        wg_x = str(df_table.loc[i, 'workgroupX'])
        wg_y = str(df_table.loc[i, 'workgroupY'])
        wg_z = str(df_table.loc[i, 'workgroupZ'])
        launch_param_key = f'{grid_x}-{grid_y}-{grid_z}-{wg_x}-{wg_y}-{wg_z}'

        if kernel_name not in kernel_name_set:
            kernel_launch_info[kernel_name] = {launch_param_key: [1]}
            kernel_name_set.add(kernel_name)
        else:
            tmp = kernel_launch_info[kernel_name]
            if tmp.get(launch_param_key) is not None:
                count = tmp[launch_param_key]
                count[0] += 1
                tmp[launch_param_key] = count
            else:
                tmp[launch_param_key] = [1]

    with open(rpd_file + f"_kernel_launch_info.csv", 'w') as csvfile:
        writer = csv.writer(csvfile)
        for cur_kernel, cur_kernel_launch_param_dict in kernel_launch_info.items():
            writer.writerow([cur_kernel] + 
                            [(f"gridX-gridY-gridZ-wgX-wgY-wgZ: {k}", f"# of calls - {cur_kernel_launch_param_dict[k][0]}") for k in cur_kernel_launch_param_dict.keys()])      

def extract_table(db, rpd_file, table_name, show_additional_metrics=False):

    """_summary_
        db: sqlite3
        rpd_file: str
        table_name: str
        show_additional_metrics: bool
    """
    
    table = pd.read_sql_query("SELECT * from %s;" % table_name, db)
    table.to_csv(f"{rpd_file}_{table_name}.csv")

    if show_additional_metrics and table_name == 'kernel':
        extract_additional_metrics(df_table=table, rpd_file=rpd_file)
        
    db.close()

def query_table(db, table_name, num_rows):
    
    """_summary_
        db: sqlite3
        table_name: str
        num_rows: int
    """
     
    cursor = db.cursor()
    cursor.execute(f"PRAGMA table_info('{table_name}')")
    table_schema = cursor.fetchall()
    table_schema_simplified = [(col[1], col[2]) for col in table_schema]
    cursor.execute(f"SELECT * FROM {str(table_name)};")
    queried_table = cursor.fetchmany(size=num_rows)
    print(f"\n|{table_name}| -> schema: {table_schema_simplified}")
    
    for i, row in enumerate(queried_table):
        print(f"\n--- Row {i + 1} --- \n{row}")

    cursor.close()
    db.close()

def main():

    SHOW_ADD_METRICS = True # if we are dumping any other table other than "kernel", we need to set this to False
    NUM_ROWS_TO_SHOW = 5

    parser = argparse.ArgumentParser(description="Process RPD file for kernel execution variance")
    parser.add_argument(
        "--input",
        type=str,
        default="",
        help="Input RPD file",
        required=True,
    )
    # parser.add_argument(
    #     "--show-schema",
    #     help="Get all tables schema info",
    #     action='store_true',
    #     required=False,
    # )
    parser.add_argument(
        "--query-table",
        help="Queries an entire table given a table name",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--extract-table",
        help="Convert an entire table to csv given a connected db",
        type=str,
        required=False,
    )

    args = parser.parse_args()
    rpd_file = args.input

    try:
        print(f"*** Connecting to {str(rpd_file)}... ***")
        db = sqlite3.connect(rpd_file)
        print("*** DB connected! ***")
    except Exception as e:
        print(e)
        exit(1)

    if args.query_table:
        print(f"*** Querying table - {str(args.query_table)} ... ***")
        query_table(db=db, 
                    table_name=str(args.query_table), 
                    num_rows=NUM_ROWS_TO_SHOW)
    
    if args.extract_table:
        print(f"*** Extracting table - {str(args.extract_table)} - from DB - {str(rpd_file)} ... ***")
        extract_table(db=db,
                      rpd_file=str(rpd_file),
                      table_name=str(args.extract_table),
                      show_additional_metrics=SHOW_ADD_METRICS)


if __name__=="__main__":
    main()