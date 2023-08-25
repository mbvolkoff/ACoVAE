import pickle
from typing import List, TextIO, BinaryIO, Type, Tuple

import numpy as np
import multiprocessing as mp
from CGRtools import SMILESRead


class SMILESParser:
    def __init__(self, symbols_dict=None, max_length=150, header=False, smiles_copies=1, freeze_charset=False):
        if symbols_dict is None:
            symbols_dict = dict()
        self.symbols_dict = symbols_dict
        self.inverse_symbols_dict = dict()
        if symbols_dict:
            self.generate_inverse_symbols_dict()
        else:
            self.symbols_dict['<PAD>'] = 0
            self.symbols_dict['<EOS>'] = 1
        self.max_length = max_length
        self.header = header
        self.double_chars_encode = {"Cl": "X", "Br": "Y", "Si": "A", "Se": "Z", "se": "z", "As": "Q", "[N+]": "W",
                                    "[O-]": "V", "[nH]": "n", "[n+]": "w", "[C@H]": "C", "[C@@H]": "C",
                                    "[C@@]": "C", "[C@]": "C"}
        self.double_chars_decode = {"X": "Cl", "Y": "Br", "A": "Si", "Z": "Se", "z": "se", "Q": "As", "W": "[N+]",
                                    "V": "[O-]", "w": "[n+]"}
        self.smiles_copies = smiles_copies
        self.freeze_charset = freeze_charset
        self.__n_features = 0

    def read_file(self, ifile: TextIO, batch_size: int = 5000):
        """
        Reads the SMILES file and transforms its items to integers stored in a new Numpy array.
        :param ifile: TextIO
        :return: Tuple of a Numpy array and Python list
        """
        data = []
        cond, tmp = None, None
        n_passed = 1
        if self.header:
            ifile.readline()
        if self.__n_features:
            tmp = np.zeros((batch_size, self.__n_features))
        for line in ifile:
            if line.strip():
                lparts = line.strip().split()
                res = self.fold_smiles(lparts[0])
                if res:
                    if n_passed % batch_size == 0:
                        print(f'Processed {n_passed} lines..')
                    data.append(res)
                    if self.__n_features == 0:
                        self.__n_features = int(lparts[-1].split(':')[0])
                        tmp = np.zeros((batch_size, self.__n_features))
                    for item in lparts[1:]:
                        tmp[(n_passed % batch_size)-1, int(item.split(':')[0])-1] = float(item.split(':')[1])
                    if n_passed % batch_size == 0:
                        if n_passed > batch_size:
                            cond = np.concatenate((cond, tmp), axis=0)
                        else:
                            cond = tmp
                        tmp = np.zeros((batch_size, self.__n_features))

                    n_passed += 1
                    # if (n_passed - 1) % 50000 == 0:
                    #     break
        ifile.close()

        if n_passed < (batch_size - 1):
            cond = tmp[:(n_passed-1), :]
        else:
            cond = np.concatenate((cond, tmp), axis=0)
            cond = cond[:(n_passed-1), :]
        smi_array = np.array(data, dtype=np.int64)
        print(smi_array.shape)
        print(cond.shape)
        #np.save("van_smi.npy", smi_array)
        #np.save("van_feats.npy", cond)
        return smi_array, cond

    def write_file(self, data: np.ndarray, ofile: TextIO, query_id: int = 1):
        """
        Decodes the predicted SMILES and writes it to the file. Don't forget to close the thread by yourself!
        :param data: Numpy array
        :param ofile: TextIO
        :param query_id: Query ID given in the Features file.
        """
        smiles_parser = SMILESRead.create_parser(ignore=True, remap=False)
        for molecule in data:
            smiles = self.unfold_smiles(molecule)
            if smiles:
                try:
                    smiles = smiles_parser(smiles)
                    smiles.standardize()
                    smiles.kekule()
                    if smiles.check_valence():
                        raise Exception('Valence errors!')
                    smiles.thiele()
                    ofile.write(f'{str(smiles)}\t{query_id}\n')
                except:
                    continue

    def fold_smiles(self, smiles: str) -> List:
        """
        Transforms the given SMILES string to a list of corresponding integers.
        :param smiles: str
        :return: List
        """
        result = []

        for key, value in [('[nH]', 'n'), ('[C]', 'C'), ('[S]', 'S'), ('[Se]', 'Se'), ('[se]', 'se'), ('[N]', 'N'),
                           ('[Si]', 'Si'), ('[I]', 'I'), ('[B]', 'B'), ('[Br]', 'Br'), ('[Cl]', 'Cl')]:
            smiles = smiles.replace(key, value)

        for key, value in self.double_chars_encode.items():
            smiles = smiles.replace(key, value)

        if len(smiles) > self.max_length or '[' in smiles or ']' in smiles or 'Mg' in smiles or 'Ca' in smiles or \
                'Na' in smiles:
            return result

        symbol_index = len(self.symbols_dict)
        for symbol in smiles:
            if self.freeze_charset and symbol not in self.symbols_dict:
                return []
            elif symbol not in self.symbols_dict:
                self.symbols_dict[symbol] = symbol_index
                symbol_index += 1
            result.append(self.symbols_dict[symbol])

        # <EOS> is 'End of smiles'
        result.append(self.symbols_dict['<EOS>'])

        # <PAD> is an Empty Atom
        while len(result) <= self.max_length:
            result.append(self.symbols_dict['<PAD>'])

        return result

    def unfold_smiles(self, array: np.ndarray) -> str:
        """
        Transforms the given Numpy array back to a SMILES string.
        :param array: Numpy array
        :return: str
        """
        result = []
        for item in array:
            item = int(item)
            if item == 0 or item not in self.inverse_symbols_dict or self.inverse_symbols_dict[item] == '<EOS>':
                break
            result.append(self.inverse_symbols_dict[item])

        result = ''.join(result)

        for key, value in self.double_chars_decode.items():
            result = result.replace(key, value)

        return result

    def generate_inverse_symbols_dict(self):
        for key, value in self.symbols_dict.items():
            self.inverse_symbols_dict[value] = key

    def load_symbols_dict(self, file: BinaryIO):
        """
        Loads the symbolic dictionary from the file.
        :param file: BinaryIO
        :return:
        """
        self.symbols_dict = pickle.load(file)

    def dump_symbols_dict(self, file: BinaryIO):
        """
        Dumps the symbolic dictionary to the file.
        :param file: BinaryIO
        :return:
        """
        pickle.dump(self.symbols_dict, file)

    def transform_data_for_decoder(self, array: np.ndarray) -> np.ndarray:
        """
        Takes the padded SMILES sequence represented via integers, deletes the '<EOS>' and '<PAD>' symbols,
        reverts the sequence and adds '!' in the beginning and '<EOS>' and '<PAD>' in the end.
        :param array: np.ndarray
        :return: np.ndarray
        """
        data = []
        start_symbol = max(self.symbols_dict.values()) + 1
        for mol in array:
            pure_smiles = list(filter(lambda s: s != self.symbols_dict['<EOS>'] and s != self.symbols_dict['<PAD>'], mol))
            pure_smiles = [start_symbol] + pure_smiles[::-1] + [self.symbols_dict['<EOS>']]
            while len(pure_smiles) < (self.max_length + 2):
                pure_smiles.append(self.symbols_dict['<PAD>'])
            data.append(pure_smiles)

        return np.array(data, dtype=np.int64)

    def parse_features(self, ifile: TextIO) -> np.ndarray:
        """
        This method reads the features file given in SVM format, where the first column is the query ID.
        :param ifile: Input features file.
        :return: Numpy array with query ID in the first column.
        """
        data = []
        for query in ifile:
            if query.strip():
                sqline = query.strip().split()
                query_id = float(sqline[0])
                tmp = [0.0] * self.__n_features
                for item in sqline[1:]:
                    tmp[int(item.split(':')[0])-1] = float(item.split(':')[1])
                data.append([query_id] + tmp)
        return np.array(data, dtype=np.float32)

    def set_n_features(self, n_features: int):
        self.__n_features = n_features

    @property
    def n_tokens(self):
        return len(self.symbols_dict)

    @property
    def n_features(self):
        return self.__n_features

def parse_isida_descr_column(isida_col: str) -> (int, float):
    key, val = list(map(int, isida_col.split(":")))
    return key, val

def parse_descriptor_string(inp: Tuple[int, str]):
    n_features, inline = inp
    lstrip = inline.strip()
    if not lstrip:
        return None #if line is empty
    smi, *descrs = lstrip.split()
    descrs_vector = np.zeros(n_features, dtype=np.int16)
    if descrs:
        descrs_list = list(map(parse_isida_descr_column, descrs)) # parse descriptors
        for (key, val) in descrs_list:
            descrs_vector[key-1] = val
    return smi, descrs_vector

def get_nfeats(inline: str):
    lstrip = inline.strip()
    _, *descrs = lstrip.split()
    descrs_list = list(map(parse_isida_descr_column, descrs)) # parse descriptors
    *_, (n_features, _) = descrs_list
    return n_features

def read_file_proc_cache(cache: List[str], smi_parser: Type[SMILESParser]):#, n_features: int):
    with mp.Pool(mp.cpu_count()) as p:
        raw_smi, isida_vecs = list(zip(*list(filter(lambda x: x is not None, p.map(parse_descriptor_string, cache)))))
    #raw_smi, isida_vecs = list(zip(*list(filter(lambda x: x is not None, map(parse_descriptor_string(n_features), cache)))))
        folded_smiles = [smi_parser.fold_smiles(smi_string) for smi_string in raw_smi]
        smi_vecs, isida_vecs = list(zip(*filter(lambda x: x[0], list(zip(folded_smiles, isida_vecs)))))
        #smi_nparrs = list(map(np.array, smi_vecs))
        smi_nparrs = [item.astype(np.int16) for item in map(np.array, smi_vecs)]
        return smi_nparrs, isida_vecs

def update_smi_isida_lists(
        cache: List[str], 
        smi_parser: Type[SMILESParser], 
        smi_vectors: List[np.ndarray], 
        isida_descr_vecs: List[np.ndarray],
        #n_features: int
        ):
    smi_nparrs, isida_vecs = read_file_proc_cache(cache, smi_parser)#, n_features)
    smi_vectors.extend(smi_nparrs)
    isida_descr_vecs.extend(isida_vecs)

def read_file(ifile: TextIO, smi_parser: Type[SMILESParser], batch_size: int = 5000):
    if smi_parser.header:
        ifile.readline() # skipping header line
    cache = []
    smi_vectors = []
    isida_descr_vecs = []
    n_features = 0
    for n_passed, line in enumerate(ifile):
        if not n_passed:
            n_features = get_nfeats(line)
        #print(n_passed)
        cache.append((n_features, line))
        if (n_passed+1) % batch_size == 0:
            print(f'Processed {n_passed} lines..')
        #parsef = parse_descriptor_string(n_features)
        #smi, descrs_vector = parsef(line)
        #folded_smiles = smi_parser.fold_smiles(smi)
        #if folded_smiles:
        #    smi_vectors.append(np.array(folded_smiles))
        #    isida_descr_vecs.append(descrs_vector)
            #try:
            update_smi_isida_lists(cache, smi_parser, smi_vectors, isida_descr_vecs)#, n_features)
            #except ValueError:
            #    raise ValueError(f"{cache}")
            cache = []
    ifile.close()
    if cache:
        #print(cache)
        update_smi_isida_lists(cache, smi_parser, smi_vectors, isida_descr_vecs)#, n_features)
    smi_out = np.stack(smi_vectors)
    feats_out = np.stack(isida_descr_vecs)
    print(smi_out.shape)
    print(feats_out.shape)
    #np.save("mod_smi1.npy", smi_out)
    #np.save("mod_feats1.npy", feats_out)
    return smi_out, feats_out, n_features