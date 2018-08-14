import os
import random
import pickle
import shutil
import fire
import glob


class SplitDataset:
    def __init__(self):
        pass

    def run(self, ind=0, numworkers=1):
        __cache_path = '/data/private/deepface/resnet_train/filelist_0813_bbox.pkl'
        __parent_dir = '/data/public/rw/datasets/faces/vggface2/'
        __original_dir = os.path.join(__parent_dir, 'train')
        __train_dir = os.path.join(__parent_dir, 'train_split2')
        __val_dir = os.path.join(__parent_dir, 'validation_split2')

        if os.path.exists(__cache_path):
            with open(__cache_path, 'rb') as f:
                d = pickle.load(f)
                filelist = d['filelist']
            print('loaded from cache file.')
        else:
            filepath = os.path.join(__original_dir, '*/*.jpg')
            filelist = glob.glob(filepath)

        counter = 0
        num_train = 0
        num_val = 0

        seg_start = ind * int(len(filelist) / numworkers)
        seg_end = (ind + 1) * int(len(filelist) / numworkers)

        eval = []
        train = []

        for file in filelist[seg_start:seg_end]:
            counter += 1
            if counter % 500 == 0:
                print('%d of %d' % (counter, len(filelist)))

            if random.uniform(0, 1) <= 0.05:
                # self.copy_src_to_dest(file, __val_dir)
                eval.append(file)
                num_val += 1
            else:
                # self.copy_src_to_dest(file, __train_dir)
                train.append(file)
                num_train += 1
        with open('/data/private/deepface/resnet_train/vggface2_eval_list.pkl', 'wb') as f:
            pickle.dump({
                'filelist': eval
            }, f, protocol=2)
        with open('/data/private/deepface/resnet_train/vggface2_train_list.pkl', 'wb') as f:
            pickle.dump({
                'filelist': train
            }, f, protocol=2)
        print("Task completed with %d training (%.2f) and %d validation (%.2f) data." % (
            num_train, num_train / len(filelist) * numworkers, num_val, num_val / len(filelist) * numworkers))

    def copy_src_to_dest(self, src, dest):
        class_dir = os.path.join(dest, os.path.basename(os.path.dirname(src)))
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
        imgname = os.path.basename(src)
        new_file_path = os.path.join(class_dir, imgname)
        shutil.copy(src, new_file_path)


if __name__ == '__main__':
    fire.Fire(SplitDataset)
