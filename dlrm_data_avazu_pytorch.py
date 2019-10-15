import sqlite3 as sq3
import torch
from torch.utils import data
import numpy as np

class AvazuDataset(data.Dataset):

    def __init__(self, db_path, split="train", dup_to_mem=True,
            chunk_size=3000000):
        self.db_conn = sq3.connect(db_path)

        if (dup_to_mem):
            on_disk_conn = self.db_conn
            self.db_conn = sq3.connect(":memory:")

            # copy on disk db into memory
            on_disk_conn.backup(self.db_conn)
            on_disk_conn.close()

        self.db_cursor = self.db_conn.cursor()

        total_items = self.db_cursor.execute("""SELECT Count(*) FROM data_cleaned""").fetchone()[0]

        #per segment training -- just to start
        indicies = np.arange(total_items)

        #total randomization
        indicies = np.random.permutation(indicies)

        #split into five segments
        indicies = np.array_split(indicies, 5)

        #randomize each segment
        #for i in range(len(indicies)):
        #    indicies[i] = np.random.permutation(indicies[i])

        #train on the first 4 segements
        train_indicies = np.concatenate(indicies[:-1])

        #split the last segment into val and test
        test_indicies = indicies[-1]
        val_indicies, test_indicies = np.array_split(test_indicies, 2)


        #we know the number of cols is small enough
        raw_from_db = list(self.db_cursor.execute("""SELECT count FROM col_counts"""))

        self.m_den = 1  #1 dense feature
        self.counts = np.array(raw_from_db).flatten()
        self.n_emb = len(self.counts)

        print(f"Avazu opened: sparse = {self.n_emb}, dense = {self.m_den}")

        if split == 'train':
            self.samples_index_lookup = train_indicies
        if split == 'val':
            self.samples_index_lookup = val_indicies
        if split == 'test':
            self.samples_index_lookup = test_indicies

        self.chunk_size = chunk_size
        self.data_chunk = None
        self.chunk_range = None

        self.load_chunk(0)

    #
    # load [start, start+chunk_size) to local storage
    def load_chunk(self, start):

        print(f"INFO: Loading chunk @ {start}")

        self.data_chunk = []
        end = self.chunk_size + start
        total_to_convert = end - start
        for ii, target_row in enumerate(self.samples_index_lookup[start:end]):

            # log every million
            if (ii % 1000000==0):
                print(f"chunk loading {start} @ {ii} / {total_to_convert}")

            #sqlite rows are indexed from 1
            sql_raw_row = target_row + 1

            data_tuple = self.db_cursor.execute(f"""SELECT * FROM data_cleaned
                                                   WHERE rowid = {sql_raw_row}
                                                """).fetchone()

            X_int = torch.tensor((data_tuple[2], ), dtype=torch.float)
            X_cat = torch.tensor(data_tuple[3:], dtype=torch.long)
            y = torch.tensor(data_tuple[1], dtype=torch.float)

            self.data_chunk.append((X_int, X_cat, y))

        self.chunk_range = (start,end)


    def __len__(self):
        return len(self.samples_index_lookup)

    def __getitem__(self, index):


        chunk_start = self.chunk_range[0]
        chunk_end = self.chunk_range[1]

        #test our current chunk
        if isinstance(index, slice):
            if (index.end - index.start) > self.chunk_size:
                raise IndexError("Slice range is larger than chunk size")

            if index.start < chunk_start or index.end >= chunk_end:
                self.load_chunk(index.start)
        else:
            if (index >= chunk_end or index < chunk_start):
                self.load_chunk(index)

        # grab new in case they were updated by the previous
        chunk_start = self.chunk_range[0]
        chunk_end = self.chunk_range[1]

        #correct indexes
        rel_index = None
        if isinstance(index, slice):
            rel_index = slice(index.start-chunk_start, index.end-chunk_end)
        else:
            rel_index = index - chunk_start

        return self.data_chunk[rel_index]


