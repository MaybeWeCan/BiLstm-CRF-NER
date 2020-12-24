
import os
from multiprocessing import Pool
from utils import load_vocabulary
from Config import Config

'''
    多线程的尝试使用,此处不考虑内存问题:
        多处理，单一写入
'''

def safe_readline(f):
    pos = f.tell()
    while True:
        try:
            return f.readline()
        except UnicodeDecodeError:
            pos -= 1
            f.seek(pos)

def process_line_to_index(line,word_to_index):

    line = line.strip().replace("\n", '')
    seq = []

    for word in line.split(" "):

        try:
            index = word_to_index[word]
        except:
            index = word_to_index["[UNK]"]

        seq.append(index)

    return seq



def async_kd_tokenizer(filename, word_to_index,worker_id, num_workers):

    with open(filename, 'r', encoding="utf-8" ) as f:
        size = os.fstat(f.fileno()).st_size  # 指针操作，所以无视文件大小
        print(f'size {size}')

        chunk_size = size // num_workers
        offset = worker_id * chunk_size
        end = offset + chunk_size

        if end > size:
            end = size - 1

        # 操作文件游标移动
        f.seek(offset)
        print(f'offset {offset}')
        print("Process in",os.getpid())

        if offset > 0:
            safe_readline(f)  # drop first incomplete line

        lines = []
        line = f.readline()

        while line:

            line = process_line_to_index(line,word_to_index)

            if not line:
                line = f.readline()
                continue

            lines.append(line)

            if f.tell() > end:
                break
            line = f.readline()
        # return lines[1:-1]
        return lines


def process_single_file(path, word_to_index, workers=4):

    assert os.path.exists(path)

    results = []
    workers_thread = []
    pool = Pool(processes=workers)

    for i in range(workers):
        w = pool.apply_async(
            async_kd_tokenizer,
            (path, word_to_index, i, workers),
        )
        workers_thread.append(w)
    pool.close()
    pool.join()

    # last output
    for w in workers_thread:
        result = w.get()
        results += result

    return results


def save_results_to_file(results, write_path):

    # write into file
    with open(write_path,"w") as wf:

        for single_seq in results:
            # "|".join(str(i) for i in data)
            wrtie_line = " ".join(str(i) for i in single_seq)
            wrtie_line += "\n"
            wf.writelines(wrtie_line)

def save_data_info(seq_results,info_txt_file):
    seq_length = len(seq_results)

    with open(info_txt_file,"w",encoding="utf-8") as f:
        f.write(str(seq_length))



if __name__ == '__main__':


    Configs = Config()

    # load vocab
    w2i_char, i2w_char = load_vocabulary(Configs.char_vocab_path)
    w2i_bio, i2w_bio = load_vocabulary(Configs.label_vocab_path)

    # process data
    seq_results = process_single_file(Configs.seq_path, w2i_char, Configs.workers)
    label_results = process_single_file(Configs.label_path, w2i_bio, Configs.workers)

    # valid datasets
    assert len(seq_results) == len(label_results)
    assert all(len(input_seq) == len(output_seq) for input_seq, output_seq in zip(seq_results, label_results))

    save_data_info(seq_results, Configs.info_txt)

    save_results_to_file(seq_results,Configs.write_seq_path)
    save_results_to_file(label_results,Configs.write_label_path)

    print("Prepare data Successfully!")


