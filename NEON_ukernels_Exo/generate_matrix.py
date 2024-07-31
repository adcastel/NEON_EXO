#/bin/python


import argparse
import math


def how(M,N, lane, arch):
    if arch == "NEON":
        if M % lane != 0:
            M = math.ceil(M/lane)*lane
        if N % lane != 0:
            N = math.ceil(N/lane)*lane

    reg_a = M//lane if M % lane == 0 else M//lane + 1
    reg_b = N if arch == "RISCV" else N//lane
    reg_c = N *  (M//lane if M % lane == 0 else M//lane + 1)
    
    return reg_a + reg_b + reg_c

def gettype(prec):
    data = ""
    if prec == "fp32" or prec == "f32":
        data = "float"
        prec = "fp32"
    elif prec == "fp16" or prec == "f16":
        data = "_Float16"
        prec = "fp16"
    elif prec == "i16":
        data = "int16_t"
    elif prec == "i8":
        data = "int8_t"
    elif prec == "i32":
        data = "int"
    else:
        print(f"Error! {prec} data type not supported!")
    return data, prec

def generate_file(MR, NR, LANE, arch, precA, precB, precC ,dest):
    dataA, precA = gettype(precA)
    dataB, precB = gettype(precB)
    dataC, precC = gettype(precC)

    if dataA == "" or dataB == "" or dataC == "":
        print("Error in datatypes")
        return
    
    with open("{}/exo_matrix_{}_{}.h".format(dest,arch, precC), 'w') as f:
        f.write(f"#include \"kernels_{arch}_{MR}x{NR}_{precC}.h\"\n")
        f.write(f"#include <stdlib.h>\n")
        f.write(f"typedef void (*ukrFunction)( void *ctxt, int_fast32_t KC, const {dataA}* alpha, {dataA} * A, int lda , {dataB} * B, int ldb, const {dataC}* beta, {dataC} *C, int ldc);\n")
        f.write(f"ukrFunction**** allocateMatrix();\nvoid fillMatrix(ukrFunction**** matrix);\nvoid freeMatrix(ukrFunction**** matrix);\n")



    with open("{}/exo_matrix_{}_{}.c".format(dest,arch, precC), 'w') as f:
        f.write(f"#include \"exo_matrix_{arch}_{precC}.h\"\n")
        f.write(f"\n")

        #allocatefunction
        tab ="    "
        f.write("ukrFunction**** allocateMatrix() {\n")
        f.write("{}ukrFunction**** matrix = (ukrFunction****)malloc({} * sizeof(ukrFunction***));\n".format(tab,MR+1))
        f.write("{}for (int i = 0; i < {}; i++) {{\n".format(tab, MR+1))
        f.write("{}matrix[i] = (ukrFunction***)malloc({} * sizeof(ukrFunction**));\n".format(tab*2,NR+1))
        f.write("{}for (int j = 0; j < {}; j++) {{\n".format(tab*2,NR+1))
        f.write("{}matrix[i][j] = (ukrFunction**)malloc({} * sizeof(ukrFunction*));\n".format(tab*3,2))
        f.write("{}for (int b = 0; b < {}; b++) {{\n".format(tab*3,2))
        f.write("{}matrix[i][j][b] = (ukrFunction*)malloc({} * sizeof(ukrFunction));\n".format(tab*4,1))
        f.write("{}}}\n".format(tab*3))
        f.write("{}}}\n".format(tab*2))
        f.write("{}}}\n".format(tab*1))
        f.write("{}return matrix;\n".format(tab))
        f.write("}\n")
        f.write(f"\n")
        f.write(f"\n")

        #fill
        f.write("void fillMatrix(ukrFunction**** matrix) {\n")
        for m in range(0,MR+1):
            for n in range(0,NR+1):
                for b in range(0,2):
                    if m == 0 or n == 0: # or how(m, n, LANE, arch) > 50:
                        f.write("{}*matrix[{}][{}][{}] = \t(ukrFunction)NULL;\n".format(tab,m,n,b))
                    else:
                        f.write("{}*matrix[{}][{}][{}] = \t(ukrFunction)gemm_{}_{}x{}_b{}_col_{};\n".format(tab,m,n,b,arch,m,n,b,precC))

        f.write("}\n")
        f.write(f"\n")
        f.write(f"\n")


        #free function
        f.write("void freeMatrix(ukrFunction**** matrix) {\n")
        f.write("{}for (int i = 0; i < {}; i++) {{\n".format(tab,MR+1))
        f.write("{}for (int j = 0; j < {}; j++) {{\n".format(tab*2,NR+1))
        f.write("{}for (int b = 0; b < {}; b++) {{\n".format(tab*3,2))
        f.write("{}free(matrix[i][j][b]);\n".format(tab*4))
        f.write("{}}}\n".format(tab*3))
        f.write("{}free(matrix[i][j]);\n".format(tab*3))
        f.write("{}}}\n".format(tab*2))
        f.write("{}free(matrix[i]);\n".format(tab*2))
        f.write("{}}}\n".format(tab))
        f.write("{}free(matrix);\n".format(tab))
        f.write("}\n")
        f.write(f"\n")
        f.write(f"\n")


def main():
    # Create ArgumentParser object
    parser = argparse.ArgumentParser(description="Receive 2 integers and 2 strings.")
    
    # Adding arguments
    parser.add_argument('MR', type=int, help='MR')
    parser.add_argument('NR', type=int, help='NR')
    parser.add_argument('LANE', type=int, help='LANE')
    parser.add_argument('arch', type=str, help='arch')
    parser.add_argument('precA', type=str, help='precision')
    parser.add_argument('precB', type=str, help='precision')
    parser.add_argument('precC', type=str, help='precision')
    #parser.add_argument('dest', type=str, help='path')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Extracting values
    MR = args.MR
    NR = args.NR
    LANE = args.LANE
    arch = args.arch.upper()
    if arch not in "RVV NEON":
        print("Arch {} not supported yet!".format(arch))
    
    precA = args.precA.lower()
    precB = args.precB.lower()
    precC = args.precC.lower()
    #dest = args.dest
    dest=f"kernels/{arch}/{MR}x{NR}"
    if gettype(precA) == "" or gettype(precB) == "" or gettype(precC) == "":
        print("Error data type")
    generate_file(MR, NR, LANE, arch, precA, precB, precC ,dest)

if __name__ == "__main__":
    main()




