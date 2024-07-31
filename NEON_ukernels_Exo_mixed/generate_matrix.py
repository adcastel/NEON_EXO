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



def generate_file(MR, NR, LANE, arch, prec):
    with open("exo_matrix_{}_{}.c".format(arch, prec), 'w') as f:
        f.write(f"#include \"exo_matrix.h\"\n")
        f.write(f"\n")

        #allocatefunction
        tab ="    "
        f.write("ukrFunction**** allocateMatrix() {\n")
        f.write("{}ukrFunction**** matrix = (ukrFunction****)malloc({} * sizeof(ukrFunction***));\n".format(tab,MR))
        f.write("{}for (int i = 0; i < {}; i++) {{\n".format(tab, MR))
        f.write("{}matrix[i] = (ukrFunction***)malloc({} * sizeof(ukrFunction**));\n".format(tab*2,MR))
        f.write("{}for (int j = 0; j < {}; j++) {{\n".format(tab*2,NR))
        f.write("{}matrix[i][j] = (ukrFunction**)malloc({} * sizeof(ukrFunction*));\n".format(tab*3,NR))
        f.write("{}for (int b = 0; b < {}; b++) {{\n".format(tab*3,2))
        f.write("{}matrix[i][j][b] = (ukrFunction*)malloc({} * sizeof(ukrFunction));\n".format(tab*4,2))
        f.write("{}}}\n".format(tab*3))
        f.write("{}}}\n".format(tab*2))
        f.write("{}}}\n".format(tab))
        f.write("{}return matrix;\n".format(tab))
        f.write("}\n")
        f.write(f"\n")
        f.write(f"\n")

        #fill
        f.write("void fillMatrix(ukrFunction**** matrix) {\n")
        for m in range(0,MR+1):
            for n in range(0,NR+1):
                for b in range(0,2):
                    if m == 0 or n == 0 or how(m, n, LANE, arch) > 50:
                        f.write("{}*matrix[{}][{}][{}] = \t(ukrFunction)NULL;\n".format(tab,m,n,b))
                    else:
                        f.write("{}*matrix[{}][{}][{}] = \t(ukrFunction)gemm_{}_{}x{}_b{}_col_{};\n".format(tab,m,n,b,arch,m,n,b,prec))

        f.write("}\n")
        f.write(f"\n")
        f.write(f"\n")


        #free function
        f.write("void freeMatrix(ukrFunction**** matrix) {\n")
        f.write("{}for (int i = 0; i < {}; i++) {{\n".format(tab,MR))
        f.write("{}for (int j = 0; j < {}; j++) {{\n".format(tab*2,NR))
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
    parser.add_argument('--MR', type=int, help='MR')
    parser.add_argument('--NR', type=int, help='NR')
    parser.add_argument('--LANE', type=int, help='LANE')
    parser.add_argument('--arch', type=str, help='arch')
    parser.add_argument('--prec', type=str, help='precision')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Extracting values
    MR = args.MR
    NR = args.NR
    LANE = args.LANE
    arch = args.arch.upper()
    if arch not in "RVV NEON":
        print("Arch {} not supported yet!".format(arch))
    
    prec = args.prec.lower()
        
    generate_file(MR, NR, LANE, arch, prec)

if __name__ == "__main__":
    main()




