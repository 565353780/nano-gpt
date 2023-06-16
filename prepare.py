from nano_gpt.Pre.shakespeare_char.prepare import prepare as \
    prepare_shakespeare_char
from nano_gpt.Pre.shakespeare.prepare import prepare as prepare_shakespeare
from nano_gpt.Pre.openwebtext.prepare import prepare as prepare_openwebtext

if __name__ == "__main__":
    shapespeare_char_dataset_folder_path = \
        '/home/chli/chLi/nanoGPT/shakespeare_char/'
    shapespeare_dataset_folder_path = \
        '/home/chli/chLi/nanoGPT/shakespeare/'
    openwebtext_dataset_folder_path = \
        '/home/chli/chLi/nanoGPT/openwebtext/'

    prepare_shakespeare_char(shapespeare_char_dataset_folder_path)
    prepare_shakespeare(shapespeare_dataset_folder_path)
    prepare_openwebtext(openwebtext_dataset_folder_path)
