
import os
import pandas as pd

def exportPandasChunks(split=2, directoryPath="./Databases/pickle_chunks/", compressionType="zstd"):
    count = 1
    for i in range(0, len(df)-1, (len(df)//split)):
        print("Exporting Chunk from " + str(i) + " to " +  str(i+(len(df)//split)+1))
        df[i:(i+(len(df)//split)+1)].to_pickle("{0}/chuck{1}.pickle".format(directoryPath, count), compression=compressionType)
        count = count + 1


def importPandasChunks(directoryPath="./Databases/pickle_chunks/", compressionType="zstd"):
    combined_df = pd.DataFrame()
    count = 0
    # Search directory code from: https://www.geeksforgeeks.org/how-to-iterate-over-files-in-directory-using-python/#
    for filename in os.scandir(directoryPath):
        if filename.is_file():
            print(filename.name)
            df_chunk = pd.read_pickle(filename.path, compression=compressionType)
            count = count + len(df_chunk)
            combined_df = pd.concat([combined_df, df_chunk], axis=0)
    
    print("Chunked Dataframes Rows: " + str(count))
    print("Combined Dataframe Rows: " + str(len(combined_df)))

    return combined_df



df = pd.read_pickle("./imageDB.pickle", compression = "zstd")
exportPandasChunks()
combined_df = importPandasChunks()
print(len(combined_df))
