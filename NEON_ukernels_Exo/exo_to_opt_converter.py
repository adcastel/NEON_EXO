import argparse


def replace_words_in_file(file_path, new_file_path, replacements):
    try:
        with open(file_path, 'r') as file:
            content = file.read()
            
            for old_word, new_word in replacements.items():
                content = content.replace(old_word, new_word)
                
        with open(new_file_path, 'w') as file:
            file.write(content)
            
        print("Words replaced successfully!")
    
    except FileNotFoundError:
        print("File not found.")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replace words in a file.")
    parser.add_argument("file_path", help="Path to the file")
    parser.add_argument("new_file_path", help="Path to the file")
    parser.add_argument("esp", help="1 or 0")
    parser.add_argument("MR", help="mr")
    parser.add_argument("NR", help="nr")
    parser.add_argument("prec", help="prec")
    parser.add_argument("arch", help="arch")
    args = parser.parse_args()
    esp = int(args.esp)
    if args.MR == "24" and args.NR=="24" and "32" in args.prec:
        esp = 0
    if args.MR == "48" and args.NR=="48" and "16" in args.prec:
        esp = 0
    
    if esp == 1:
        lda = args.MR
        ldb = args.NR
    else:
        lda = "lda"
        ldb = "ldb"
    
    replacements = {
            "float32x4_t B_reg[1]":"float32x4_t B_reg_0;",
            "float32x4_t B_reg[2]":"float32x4_t B_reg_0, B_reg_1;",
            "float32x4_t B_reg[3]":"float32x4_t B_reg_0, B_reg_1, B_reg_2;",
            "float32x4_t B_reg[4]":"float32x4_t B_reg_0, B_reg_1, B_reg_2, B_reg_3;",
            "float32x4_t B_reg[5]":"float32x4_t B_reg_0, B_reg_1, B_reg_2, B_reg_3, B_reg_4;",
            "float32x4_t B_reg[6]":"float32x4_t B_reg_0, B_reg_1, B_reg_2, B_reg_3, B_reg_4, B_reg_5;",
            "float16x8_t B_reg[1]":"float16x8_t B_reg_0;",
            "float16x8_t B_reg[2]":"float16x8_t B_reg_0, B_reg_1;",
            "float16x8_t B_reg[3]":"float16x8_t B_reg_0, B_reg_1, B_reg_2;",
            "float16x8_t B_reg[4]":"float16x8_t B_reg_0, B_reg_1, B_reg_2, B_reg_3;",
            "float16x8_t B_reg[5]":"float16x8_t B_reg_0, B_reg_1, B_reg_2, B_reg_3, B_reg_4;",
            "float16x8_t B_reg[6]":"float16x8_t B_reg_0, B_reg_1, B_reg_2, B_reg_3, B_reg_4, B_reg_5;",
            "int32x4_t B_reg[1]":"int32x4_t B_reg_0;",
            "int32x4_t B_reg[2]":"int32x4_t B_reg_0, B_reg_1;",
            "int32x4_t B_reg[3]":"int32x4_t B_reg_0, B_reg_1, B_reg_2;",
            "int32x4_t B_reg[4]":"int32x4_t B_reg_0, B_reg_1, B_reg_2, B_reg_3;",
            "int32x4_t B_reg[5]":"int32x4_t B_reg_0, B_reg_1, B_reg_2, B_reg_3, B_reg_4;",
            "int32x4_t B_reg[6]":"int32x4_t B_reg_0, B_reg_1, B_reg_2, B_reg_3, B_reg_4, B_reg_5;",
            "int16x8_t B_reg[1]":"int16x8_t B_reg_0;",
            "int16x8_t B_reg[2]":"int16x8_t B_reg_0, B_reg_1;",
            "int16x8_t B_reg[3]":"int16x8_t B_reg_0, B_reg_1, B_reg_2;",
            "int16x8_t B_reg[4]":"int16x8_t B_reg_0, B_reg_1, B_reg_2, B_reg_3;",
            "int16x8_t B_reg[5]":"int16x8_t B_reg_0, B_reg_1, B_reg_2, B_reg_3, B_reg_4;",
            "int16x8_t B_reg[6]":"int16x8_t B_reg_0, B_reg_1, B_reg_2, B_reg_3, B_reg_4, B_reg_5;",
            "B_reg[0]": "B_reg_0",
            "B_reg[1]": "B_reg_1",
            "B_reg[2]": "B_reg_2",
            "B_reg[3]": "B_reg_3",
            "B_reg[4]": "B_reg_4",
            "B_reg[5]": "B_reg_5",
            "A.data": "A",
            "B.data": "B",
            "C.data": "C",
            "Ci.data": "Ci",
            #"B.strides[0]": "ldb",
            #"A.strides[0]": "lda",
            "C.strides[0]": "ldc",
            "Ci.strides[0]": "ldci",
            "B.strides[1]": "1",
            "A.strides[1]": "1",
            "C.strides[1]": "1",
            "Ci.strides[1]": "1",
            "struct exo_win_2f32c B": "float * B, int ldb",
            "struct exo_win_2f32c A": "float * A, int lda",
            "struct exo_win_2f32 Ci": "float * Ci, int ldci",
            "struct exo_win_2f32 C": "float * C, int ldc",
            "struct exo_win_2f16c B": "_Float16 * B, int ldb",
            "struct exo_win_2f16c A": "_Float16 * A, int lda",
            "struct exo_win_2f16 Ci": "_Float16 * Ci, int ldci",
            "struct exo_win_2f16 C": "_Float16 C, int ldc",
            "struct exo_win_2i32c B": "int * B, int ldb",
            "struct exo_win_2i32c A": "int * A, int lda",
            "struct exo_win_2i32 Ci": "int * Ci, int ldci",
            "struct exo_win_2i32 C": "int * C, int ldc",
            "struct exo_win_2i16c B": "int16_t * B, int ldb",
            "struct exo_win_2i16c A": "int16_t * A, int lda",
            "struct exo_win_2i16 Ci": "int16_t * Ci, ldci",
            "struct exo_win_2i16 C": "int16_t * C, int ldc",
            "struct exo_win_2i8c B": "int8_t * B, int ldb",
            "struct exo_win_2i8c A": "int8_t * A, int lda",
            "struct exo_win_2i8 Ci": "int8_t * Ci, int ldci",
            "struct exo_win_2i8 C": "int8_t * C, int ldc",
            # Add more word replacements as needed
            }
    replacements["B.strides[0]"] = ldb
    replacements["A.strides[0]"] = lda
    
    replace_words_in_file(args.file_path, args.new_file_path, replacements)
