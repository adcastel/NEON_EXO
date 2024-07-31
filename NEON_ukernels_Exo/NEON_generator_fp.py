from __future__ import annotations
from exo import *
from exo import proc
from exo.platforms.neon import *
from exo.stdlib.scheduling import *
import math

@instr("{dst_data} = vld1q_f32(&{src_data});")
def neon_vld_4xf32(dst: [f32][4] @ Neon, src: [f32][4] @ DRAM, e: index):
    assert stride(src, 0) == 1
    assert stride(dst, 0) == 1
    assert e > 0
    assert e <= 4
    for i in seq(0, e):
        dst[i] = src[i]

@instr("vst1q_f32(&{dst_data}, {src_data});")
def neon_vst_4xf32(dst: [f32][4] @ DRAM, src: [f32][4] @ Neon, e: index):
    assert stride(src, 0) == 1
    assert stride(dst, 0) == 1
    assert e > 0
    assert e <= 4
    
    for i in seq(0, e):
        dst[i] = src[i]

@instr("{dst_data} = vld1q_f16(&{src_data});")
def neon_vld_8xf16(dst: [f16][8] @ Neon, src: [f16][8] @ DRAM, e: index):
    assert stride(src, 0) == 1
    assert stride(dst, 0) == 1
    assert e > 0
    assert e <= 8
    for i in seq(0, e):
        dst[i] = src[i]

@instr("vst1q_f16(&{dst_data}, {src_data});")
def neon_vst_8xf16(dst: [f16][8] @ DRAM, src: [f16][8] @ Neon, e: index):
    assert stride(src, 0) == 1
    assert stride(dst, 0) == 1
    assert e > 0
    assert e <= 8
    
    for i in seq(0, e):
        dst[i] = src[i]

@instr("{dst_data} = vfmaq_laneq_f32({dst_data}, {lhs_data}, {rhs_data}, {lane});")
def dyn_neon_vfmla_4xf32_4xf32(
        dst: [f32][4] @ Neon, lhs: [f32][4] @ Neon, rhs: [f32][4] @ Neon, lane: index, 
        e: index):
    assert stride(dst, 0) == 1
    assert stride(lhs, 0) == 1
    assert stride(rhs, 0) == 1
    assert lane >= 0
    assert lane < 4
    assert e >= 0
    assert e <= 4
    
    for i in seq(0, e):
        dst[i] += lhs[i] * rhs[lane]


# WORKS GREAT FOR FMLA BUT WE NEED TO MANUAL UNROLL B ONCE THE CODE IS GENERATED
@instr("{dst_data} = vfmaq_laneq_f32({dst_data}, {lhs_data}, {rhs_data}, {jtt});")
def neon_vfmla_4xf32_4xf32(
        dst: [f32][4] @ Neon, lhs: [f32][4] @ Neon, rhs: [f32][1,4] @ Neon, 
        jtt:index,
        ):
    assert stride(dst, 0) == 1
    assert stride(lhs, 0) == 1
    assert stride(rhs, 0) == 1
    assert jtt >= 0
    assert jtt < 4
    
    for i in seq(0, 4):
        dst[i] += lhs[i] * rhs[0,jtt]

@instr("{dst_data} = vfmaq_laneq_f16({dst_data}, {lhs_data}, {rhs_data}, {jtt});")
def neon_vfmla_8xf16_8xf16(
        dst: [f16][8] @ Neon, lhs: [f16][8] @ Neon, rhs: [f16][1,8] @ Neon, 
        jtt:index,
        ):
    assert stride(dst, 0) == 1
    assert stride(lhs, 0) == 1
    assert stride(rhs, 0) == 1
    assert jtt >= 0
    assert jtt < 8
    
    for i in seq(0, 8):
        dst[i] += lhs[i] * rhs[0,jtt]


# WORKS GREAT FOR TAIL CASES
@instr("{dst_data} = vfmaq_laneq_f32({dst_data}, {lhs_data}, {rhs_data}, {jtt});")
def neon_vfmla_4xf32_4xf32_ori(
        dst: [f32][4] @ Neon, lhs: [f32][4] @ Neon, rhs: [f32][4] @ Neon, 
        jtt:index,
        ):
    assert stride(dst, 0) == 1
    assert stride(lhs, 0) == 1
    assert stride(rhs, 0) == 1
    assert jtt >= 0
    assert jtt < 4
    
    for i in seq(0, 4):
        dst[i] += lhs[i] * rhs[jtt]

# WORKS GREAT FOR TAIL CASES
@instr("{dst_data} = vfmaq_laneq_f16({dst_data}, {lhs_data}, {rhs_data}, {jtt});")
def neon_vfmla_8xf16_8xf16_ori(
        dst: [f16][8] @ Neon, lhs: [f16][8] @ Neon, rhs: [f16][8] @ Neon, 
        jtt:index,
        ):
    assert stride(dst, 0) == 1
    assert stride(lhs, 0) == 1
    assert stride(rhs, 0) == 1
    assert jtt >= 0
    assert jtt < 8
    
    for i in seq(0, 8):
        dst[i] += lhs[i] * rhs[jtt]

@instr("{dst_data} = vmovq_n_f16((_Float16)0.0);")
def neon_zero_8xf16_new(dst: [f16][8] @ Neon):
    assert stride(dst, 0) == 1
    
    for i in seq(0, 8):
        dst[i] = 0.0

# A TRY OF AUTOGENERATING THE ENTIRE COMPUTATION BUT IT FAILS WITH MULTIPLE FUNCTIONS IN THE SAME FILE
# HOWEVER, IF A KERNEL PER FILE IS GENERATED, THEN WE HAVE A PROBLEM WITH THE REDECLARATION OF EXO STRUCTS
def generate_fmla(p,l1,l2,l3,l4):

    inst = ""
    for jt in range(l1):
        for it in range(l2):
            for jtt in range(l3):
                #for itt in range(l4):
                creg = "C_reg_{}_{}".format(jtt+l3*jt,it)#,l4)
                areg = "A_reg_{}".format(it)
                breg = "B_reg_{}".format(jt)
                inst+="{} = vfmaq_laneq_f32({}, {}, {}, {});\n  ".format(creg,creg,areg,breg,jtt)
    
    intrin = ( inst ) 
    
    @instr(intrin)
    def my_neon_vfmla_4xf32_4xf32(
            jt: size, jtt: size, itt: size, it:size,
            dst: [f32][jtt+4*jt,it*4,4] @ Neon, lhs: [f32][it*4,4] @ Neon, rhs: [f32][jt*4,jtt] @ Neon
    ):
        assert stride(dst, 0) == 1
        assert stride(dst, 1) == 1
        assert stride(dst, 2) == 1
        assert stride(lhs, 0) == 1
        assert stride(lhs, 1) == 1
        assert stride(rhs, 0) == 1
        assert stride(rhs, 1) == 1
    
        assert itt == 4

        for b in seq(0, jt):
            for i in seq(0, it):
                for j in seq(0, jtt):
                    for a in seq(0, itt):
                        dst[j+4*b,i,a] += lhs[i,a] * rhs[b,j]

    p = replace_all(p, my_neon_vfmla_4xf32_4xf32)
    return p

# ANOTHER TRY FOR GENERATING THE ENTIRE FMLA LIST
def generate_fmla_2(p,l1,l2,l3):

    inst = ""
    for jt in range(l1):
        for jtt in range(l2):
            creg = "C_reg_{}".format(jtt+l3*jt)#,l4)
            areg = "A_reg"
            breg = "B_reg_{}".format(jt)
            inst+="{} = vfmaq_laneq_f32({}, {}, {}, {});\n  ".format(creg,creg,areg,breg,jtt)
    
    intrin = ( inst ) 
    
    @instr(intrin)
    def my_neon_vfmla_4xf32_4xf32_mr(
            jt: size, jtt: size, itt: size,
            dst: [f32][jtt+4*jt,4] @ Neon, lhs: [f32][4] @ Neon, rhs: [f32][jt*4,jtt] @ Neon
    ):
        assert stride(dst, 0) == 1
        assert stride(dst, 1) == 1
        assert stride(lhs, 0) == 1
        assert stride(rhs, 0) == 1
        assert stride(rhs, 1) == 1
        assert itt == 4

        for b in seq(0, jt):
            for j in seq(0, jtt):
                for i in seq(0, itt):
                    dst[j+4*b,i] += lhs[i] * rhs[b,j]

    p = replace_all(p, my_neon_vfmla_4xf32_4xf32_mr)
    return p

# ANOTHER ONE BUT GENERATING A PART OF IT
def generate_fmla_part(p,v1,v2,v3,LANE):

    inst = ""
    creg = "C_reg[{}, {}]".format(v3+LANE*v1,v2)#,l4)
    areg = "A_reg[{}]".format(v2)
    breg = "B_reg[{}]".format(v1)
    inst+="{} = vfmaq_laneq_f32({}, {}, {}, {});\n".format(creg,creg,areg,breg,v3)
    
    intrin = ( inst ) 
    
    @instr(intrin)
    def part_neon_vfmla_4xf32_4xf32(
            dst: [f32][4] @ Neon, lhs: [f32][4] @ Neon, rhs: [f32][4] @ Neon, e: index, lane: index
    ):
        assert stride(dst, 0) == 1
        assert stride(lhs, 0) == 1
        assert stride(rhs, 0) == 1
    
        assert lane >= 0
        assert lane < 4
        assert e >= 0
        assert e < 4
    

        for i in seq(0, e):
            dst[i] += lhs[i] * rhs[lane]

    p = replace(p, p.find('for itt in _:_'),part_neon_vfmla_4xf32_4xf32)
    return p

def generator(MR,NR, pre,EMR, ENR, beta0):
    # MR -> REAL MR value
    # NR -> REAL NR value
    
    # EMR -> NEXT MR VALUE MULTIPLE OF LANE
    # ENR -> NEXT NR VALUE MULTIPLE OF LANE

    # beta0 -> Boolean  for indicating if beta is 0.
    
    # UKERNEL BASE FOR ALL THE CASES WHERE MR AND NR ARE MULTIPLE OF LANE 
    @proc
    def ukernel_main(
            KC: size,
            MR: size,
            NR: size,
            alpha: f32[1],
            A: f32[KC, MR] @ DRAM,
            B: f32[KC, NR] @ DRAM,
            beta: f32[1],
            C: f32[NR, MR] @ DRAM,
            ):

        #assert stride(A, 0) == EMR
        #assert stride(A, 1) == 1
        #assert stride(B, 0) == ENR
        #assert stride(B, 1) == 1
        #assert stride(C, 1) == 1

        for k in seq(0, KC): 
            for j in seq(0, NR): 
                for i in seq(0, MR): 
                    C[j, i] += A[k,i] * B[k,j]

    # UKERNEL USED WHEN NR IS NOT MULTIPLE OF LANE 
    @proc
    def ukernel_edge(
            KC: size,
            MR: size,
            NR: size,
            alpha: f32[1],
            A: f32[KC, EMR] @ DRAM,
            B: f32[KC, NR] @ DRAM,
            b: f32[1],
            Ci: f32[NR, EMR] @ DRAM,
            ):
        assert NR <= ENR
        assert MR <= EMR
        assert stride(A, 1) == 1
        assert stride(B, 1) == 1
        assert stride(Ci, 1) == 1
        C : f32[ENR,EMR] @ DRAM
        
        for k in seq(0, KC): 
            for j in seq(0, NR): 
                for i in seq(0, EMR): 
                    C[j, i] += A[k,i] * B[k,j]
        
        if beta0 == True: 
            for j in seq(0, NR): 
                for i in seq(0, MR): 
                    Ci[j, i] = C[j,i]
        else:
            for j in seq(0, NR): 
                for i in seq(0, MR): 
                    Ci[j, i] += C[j,i]
    
    # UKERNEL USED WHEN MR AND NR ARE NOT MULTIPLE OF LANE 
    @proc
    def ukernel_edge_esp(
            KC: size,
            MR: size,
            NR: size,
            alpha: f32[1],
            A: f32[KC, EMR] @ DRAM,
            B: f32[KC, ENR] @ DRAM,
            b: f32[1],
            Ci: f32[NR, MR] @ DRAM,
            ):
        assert NR <= ENR
        assert MR <= EMR
        assert stride(A, 1) == 1
        assert stride(B, 1) == 1
        assert stride(Ci, 1) == 1
        C : f32[ENR,EMR] @ DRAM
        
        for k in seq(0, KC): 
            for j in seq(0, ENR): 
                for i in seq(0, EMR): 
                    C[j, i] += A[k,i] * B[k,j]
        if beta0 == True: 
            for j in seq(0, NR): 
                for i in seq(0, MR): 
                    Ci[j, i] = C[j,i]
        else:
            for j in seq(0, NR): 
                for i in seq(0, MR): 
                    Ci[j, i] += C[j,i]


    def make_tail(t,despA,despB,MR,NR,LANE):
        if NR % LANE != 0 and MR % LANE == 0:
            t = stage_mem(t, "C[_] += _", "C[{} + jtt,itt + {} *it]".format(despB,despA), "C_regt")
        t = expand_dim(t, 'C_regt', LANE, 'itt') #, unsafe_disable_checks=True)
        t = expand_dim(t, 'C_regt', MR//LANE, 'it') #, unsafe_disable_checks=True)
        t = expand_dim(t, 'C_regt', NR, '{} + jtt'.format(despB)) #, unsafe_disable_checks=True)
        t = lift_alloc(t, 'C_regt', n_lifts=4)
        t     = autofission(t, t.find('{}[_] = _'.format('C_regt')).after(), n_lifts=4)
        t = autofission(t, t.find('{}[{} + jtt, itt + {} * it] = _'.format('C',despB,despA)).before()
                , n_lifts=4)
        t = simplify(t)
        t = set_memory(t, 'C_regt', intrinsics['mem'])
        t = replace_all(t, intrinsics['load'])
        t = replace_all(t, intrinsics['store'])
        t = simplify(t)
        Buf = 'A'
        Xreg='{}_regt'.format(Buf)
        t = bind_expr(t, '{}[_] #{}'.format(Buf,MR//LANE),Xreg)
        t = simplify(t)
        t = expand_dim(t, Xreg , LANE, 'itt') #, unsafe_disable_checks=True)
        t = expand_dim(t, Xreg , MR//LANE, 'it') #, unsafe_disable_checks=True)
        t = lift_alloc(t, Xreg, n_lifts=4)
        t = autofission(t, t.find('{}[_] = _'.format(Xreg)).after(),n_lifts=3)
        t = set_memory(t, 'A_regt', intrinsics['mem'])
        t = replace(t, t.find('for itt in _:_'),intrinsics['load'])
        t = simplify(t)
        scal = 'B'
        scr = '{}_regt'.format(scal)
        t = bind_expr(t, '{}[_] #{}'.format(scal,NR//LANE),scr)
        t = expand_dim(t, scr, LANE, 'jtt') #, unsafe_disable_checks=True)
        t = simplify(t)
        try:
            t = expand_dim(t, scr, NR//LANE, 'jt') #, unsafe_disable_checks=True)
        except:
            print("jtt?")

        t = simplify(t)
        t = lift_alloc(t, scr, n_lifts=4)
        t = autofission(t, t.find('{}[_] = _'.format(scr)).after(), n_lifts=3)
        t = set_memory(t, 'B_regt', intrinsics['mem'])
        t = replace(t, t.find('for jtt in _:_ #3'),intrinsics['load'])
        t = simplify(t)
        t = replace(t, t.find('for itt in _:_'),intrinsics['fmla'])

        t = simplify(t)
    
    
        return t
    
    def reorder_up(p, stmt_pattern, n=1):
        for _ in range(n):
            c = p.find(stmt_pattern).expand(1, 0)
            p = reorder_stmts(p, c)
        return p
    
    def moveup(p,expr):
        while True:
            try:
                p = reorder_up(p, expr)
            except:
                break;
        return p
    
    def unrollbuffers(p,buf):
        p = unroll_buffer(p, buf,0) 
        return p
    
    def make_tail_ok(t):
        t = stage_mem(t, "C[_] += _", "C[j,i]".format(LANE), "C_reg", init_zero=True if beta0 or MR%LANE !=0 or NR%LANE!=0 else False)
        t = expand_dim(t, 'C_reg', LANE, 'i') #, unsafe_disable_checks=True)
        t = expand_dim(t, 'C_reg', NR, 'j') #, unsafe_disable_checks=True)
        t = lift_alloc(t, 'C_reg', n_lifts=3)
        t     = autofission(t, t.find('{}[_] = _'.format('C_reg')).after(), n_lifts=3)
        t = autofission(t, t.find('{}[j,i] = _'.format('C',LANE)).before(), n_lifts=3)
        t = simplify(t)
        t = set_memory(t, 'C_reg', intrinsics['mem'])
        t = replace_all(t, intrinsics['zeros'] if beta0 or MR%LANE!=0 or NR%LANE!=0 else intrinsics['load'])
        t = replace_all(t, intrinsics['store'])
        t = simplify(t)
        Buf = 'A'
        Xreg='{}_reg'.format(Buf)
        t = bind_expr(t, '{}[_]'.format(Buf),Xreg)
        t = simplify(t)
        t = expand_dim(t, Xreg , LANE, 'i') #, unsafe_disable_checks=True)
        t = lift_alloc(t, Xreg, n_lifts=3)
        t = autofission(t, t.find('{}[_] = _'.format(Xreg)).after(),n_lifts=2)
        t = set_memory(t, 'A_reg', intrinsics['mem'])
        t = replace(t, t.find('for i in _:_'),intrinsics['load'])
        t = simplify(t)
        scal = 'B'
        scr = '{}_reg'.format(scal)
        t = bind_expr(t,scal,scr)
        t = simplify(t)
        t = expand_dim(t, scr, LANE, 'j') #, unsafe_disable_checks=True)
        t = simplify(t)
        t = lift_alloc(t, scr, n_lifts=3)
        t = autofission(t, t.find('{}[_] = _'.format(scr)).after(), n_lifts=2)
        t = set_memory(t, 'B_reg', intrinsics['mem'])
        #print(t)
        t = replace(t, t.find('for j in _:_ #1'),intrinsics['load'])
        if pre == "fp32": 
            t = replace(t, t.find('for i in _:_'),neon_vfmla_4xf32_4xf32_ori)
        if pre == "fp16":
            t = set_precision(t, "A_reg", precision)
            t = set_precision(t, "B_reg", precision)
            t = replace(t, t.find('for i in _:_'),neon_vfmla_8xf16_8xf16_ori)
        t = simplify(t)
    
        t = unroll_loop(t, "j")
        t = unroll_loop(t, "j")
        t = unroll_loop(t, "j")
        try:
            t = unroll_loop(t, "i")
        except:
            print("There is not any i loop")
        try:
            t = unroll_loop(t, "j")
        except:
            print("There is not any j loop")
        t=unrollbuffers(t, "C_reg")
    
        return t
    
    def make_lanes(p):
        if MR//LANE >= 1:
            loop='i'
            p =  divide_loop(p,loop, LANE, ['{}t'.format(loop),'{}tt'.format(loop)], tail='cut')
        loop='j'
        p =  divide_loop(p,loop, LANE, ['{}t'.format(loop),'{}tt'.format(loop)], tail='cut')
        p = simplify(p)
        if MR//LANE >= 1:
            if MR % LANE or NR % LANE:
                p = stage_mem(p, "C[_] += _", "C[jtt + {} * jt, itt + {} * it]".format(LANE,LANE), "C_reg", init_zero=True) #REGISTERS FOR MULTIPLE OF LANE
            else:
                p = stage_mem(p, "C[_] += _", "C[jtt + {} * jt, itt + {} * it]".format(LANE,LANE), "C_reg", init_zero=beta0) #REGISTERS FOR MULTIPLE OF LANE
        else:
            if MR % LANE or NR % LANE:
                p = stage_mem(p, "C[_] += _", "C[jtt + {} * jt, i]".format(LANE), "C_reg", init_zero=True) #REGISTERS FOR MULTIPLE OF LANE
            else:
                p = stage_mem(p, "C[_] += _", "C[jtt + {} * jt, i]".format(LANE), "C_reg", init_zero=beta0) #REGISTERS FOR MULTIPLE OF LANE
        # PREPARE REGISTERS FOR MULTIPLE
        if MR//LANE >= 1:
            p = expand_dim(p, 'C_reg', LANE, 'itt') #, unsafe_disable_checks=True)
        else:
            p = expand_dim(p, 'C_reg', LANE, 'i') #, unsafe_disable_checks=True)
        if MR % LANE != 0:
            if MR//LANE >= 1:
                p = expand_dim(p, 'C_reg', EMR//LANE, 'it') #, unsafe_disable_checks=True)
            else:
                pass #p = expand_dim(p, 'C_reg', 1, 'i', unsafe_disable_checks=True)
            p = expand_dim(p, 'C_reg', ENR, 'jt * {} + jtt'.format(LANE)) #, unsafe_disable_checks=True)
        else:
            if MR//LANE >= 1:
                p = expand_dim(p, 'C_reg', MR//LANE, 'it') #, unsafe_disable_checks=True)
            else:
                pass
            p = expand_dim(p, 'C_reg', ENR, 'jt * {} + jtt'.format(LANE)) #, unsafe_disable_checks=True)
        p = lift_alloc(p, 'C_reg', n_lifts=5 if MR//LANE >= 1 else 4)
        p = simplify(p)
        
        #MOVE LOADS AND STORES OF THE CASE WHEN MULTIPLE
        p = autofission(p, p.find('{}[_] = _'.format('C_reg')).after(), n_lifts=5)
        pat = '{}[jtt + {} * jt,itt+{}*it] = _'.format('C',LANE,LANE) if MR//LANE >= 1 else '{}[jtt + {} * jt,i] = _'.format('C',LANE)
        p = autofission(p, p.find(pat).before(), n_lifts=5)
        p = simplify(p)
        
        p = set_memory(p, 'C_reg',intrinsics['mem'])
        p = replace_all(p, intrinsics['zeros'] if beta0 or MR%LANE !=0 or NR%LANE !=0 else intrinsics['load'])
        p = replace_all(p, intrinsics['store'])
        
        Buf = 'A'
        Xreg='{}_reg'.format(Buf)
        p = bind_expr(p, '{}[_]'.format(Buf),Xreg)
        p = simplify(p)
        loop = 'i'
        p = expand_dim(p, Xreg , LANE, '{}tt'.format(loop) if MR//LANE >= 1 else loop) #, unsafe_disable_checks=True)
        if MR//LANE >= 1:
            p = expand_dim(p, Xreg, MR//LANE if MR % LANE == 0 else EMR//LANE, '{}t'.format(loop)) #, unsafe_disable_checks=True)
        p = lift_alloc(p, Xreg, n_lifts=5  if MR//LANE >= 1 else 4 )
        p = autofission(p, p.find('{}[_] = _'.format(Xreg)).after(),n_lifts=4 if MR//LANE >= 1 else 3)
        p = set_memory(p, 'A_reg', intrinsics['mem'])
        p = set_precision(p,'A_reg',precision) 
        scal = 'B'
        scr = '{}_reg'.format(scal)
        p = bind_expr(p,scal,scr)
        p = expand_dim(p, scr, LANE, 'jtt') #, unsafe_disable_checks=True)
        p = simplify(p)
        p = expand_dim(p, scr, NR//LANE if NR % LANE == 0 else ENR//LANE, 'jt') #, unsafe_disable_checks=True)
        p = simplify(p)
        p = lift_alloc(p, scr, n_lifts=5  if MR//LANE >= 1 else 4)
        p = autofission(p, p.find('{}[_] = _'.format(scr)).after(), n_lifts=4 if MR//LANE >= 1 else 3)
        p = simplify(p)
        p = set_memory(p, 'B_reg', intrinsics['mem'])
        p = set_precision(p,'B_reg',precision) 
        p = simplify(p)
        # C
        if MR//LANE >= 1:
            p = unroll_loop(p,'it')
        p = unroll_loop(p,'jtt')
        p = unroll_loop(p,'jt')
        
        
        #A
        if MR//LANE >= 1:
            p = unroll_loop(p,'it')
        #B
        p = unroll_loop(p,'jt')
        
        p = replace_all(p, intrinsics['load'])
        
        if MR//LANE >= 1:
            p = reorder_loops(p,'jtt it')
        
        p = simplify(p)
        p = replace_all(p, intrinsics['fmla'])
        while True:
            try:
                p = unroll_loop(p,'it')
            except:
                break
        while True:
            try:
                p = unroll_loop(p,'jtt')
            except:
                break
        while True:
            try:
                p = unroll_loop(p,'jt')
            except:
                break
        p = simplify(p)
        p = moveup(p, "B_reg : _")
        p = moveup(p, "A_reg : _")
        p=unrollbuffers(p, "C_reg")
        for i in range(NR if NR % LANE == 0 else ENR):
            try:
                p=unrollbuffers(p, "C_reg_{}".format(i))
            except:
                break;
        try:
            p=unrollbuffers(p, "A_reg")
        except:
            print("Check {}x{}".format(MR,NR))
        try:
            p=unrollbuffers(p, "B_reg")
        except:
            print("Check {}x{}".format(MR,NR))
        return p
    if pre == "fp32":
        precision="f32"
        LANE=4
        intrinsics = {'load': neon_vld_4xf32, 'store': neon_vst_4xf32, 'fmla':  neon_vfmla_4xf32_4xf32,
            'bcast': neon_broadcast_4xf32, 'vmul':neon_vmul_4xf32, 'zeros': neon_zero_4xf32, 'mem':Neon}
        #print(intrinsics) 
    elif pre == "fp16":
        precision="f16"
        LANE=8
        intrinsics = {'load': neon_vld_8xf16, 'store': neon_vst_8xf16, 'fmla':  neon_vfmla_8xf16_8xf16,
            'bcast': neon_broadcast_8xf16, 'vmul':neon_vmul_8xf16, 'zeros': neon_zero_8xf16_new, 'mem':Neon}


    if MR <= LANE and NR <= LANE:
        if MR == LANE and NR == LANE:
            p=ukernel_main

            p = set_precision(p, "C", precision)
            p = set_precision(p, "A", precision)
            p = set_precision(p, "B", precision)
            p = set_window(p, "C", True)
            p = set_window(p, "A", True)
            p = set_window(p, "B", True)
        else:
            p=ukernel_edge
            p = set_precision(p, "Ci", precision)
            p = set_precision(p, "C", precision)
            p = set_precision(p, "A", precision)
            p = set_precision(p, "B", precision)
            p = set_window(p, "Ci", True)
            p = set_window(p, "A", True)
            p = set_window(p, "B", True)
        p = rename(p, "gemm_{}_{}x{}_b{}_{}_{}".format("NEON", MR,NR,0 if beta0 else 1, "col",pre))
        p = p.partial_eval(MR=MR, NR=NR)
        p = make_tail_ok(p)
        p = simplify(p)
        return p
    
    if MR % LANE == 0 and NR % LANE == 0:
        p=ukernel_main
        #print(precision)
        p = set_precision(p, "C", precision)
        p = set_precision(p, "A", precision)
        p = set_precision(p, "B", precision)
        p = set_precision(p, "alpha", precision)
        p = set_precision(p, "beta", precision)
        p = set_window(p, "C", True)
        p = set_window(p, "A", True)
        p = set_window(p, "B", True)
        p = rename(p, "gemm_{}_{}x{}_b{}_{}_{}".format("NEON", MR,NR,0 if beta0 else 1, "col",pre))
        p = p.partial_eval(MR=MR,NR=NR)
        p = simplify(p)
        p = make_lanes(p)
        p = simplify(p)
        #print(p)
        return p

    else:
        #if MR % LANE != 0 or NR % LANE != 0 or MR % LANE != 0 and NR % LANE == 0:
        p=ukernel_edge_esp
        p = set_precision(p, "Ci", precision)
        p = set_precision(p, "C", precision)
        p = set_precision(p, "A", precision)
        p = set_precision(p, "B", precision)
        p = set_window(p, "Ci", True)
        p = set_window(p, "A", True)
        p = set_window(p, "B", True)
        p = rename(p, "gemm_{}_{}x{}_b{}_{}_{}".format("NEON", MR,NR,0 if beta0 else 1, "col",pre))
        p = p.partial_eval(MR=MR,NR=NR)
        p = simplify(p)
        p = make_lanes(p)
        p = simplify(p)
        p = unroll_loop(p,'i')
        p = unroll_loop(p,'j')
        p = simplify(p)
        #print("FINAL",p)
        return p

    #when MR % LANE == 0 but NR % LANE != 0:
    # AQUI NUNCA LLEGA!
    if MR % LANE == 0:
        p=ukernel_edge
        p = set_window(p, "Ci", True)
    
        p = rename(p, "gemm_{}_{}x{}_{}_{}".format("NEON", MR,NR,"col","f32"))
        p = p.partial_eval(MR=MR,NR=NR)
        p = simplify(p)
        loop='i'
        p =  divide_loop(p,loop, LANE, ['{}t'.format(loop),'{}tt'.format(loop)], tail='cut') #perfect=True)
        p = simplify(p)
        loop='j'
        p =  divide_loop(p,loop, LANE, ['{}t'.format(loop),'{}tt'.format(loop)], tail='cut')
        p = simplify(p)
        if NR / LANE > 1:
            p = autofission(p, p.find("for jtt in _:_ #1").before(),n_lifts=2)
        #else:
        #    p = autofission(p, p.find("for jt in _:_ #1").before(),n_lifts=1)
        if NR / LANE > 1:
            p = stage_mem(p, "C[_] += _", "C[jtt + 4 * jt, itt + {} * it]".format(LANE,LANE), "C_reg") #REGISTERS FOR MULTIPLE OF LANE
        else:
            p = stage_mem(p, "C[_] += _", "C[jtt, itt + {} * it]".format(LANE), "C_reg") #REGISTERS FOR MULTIPLE OF LANE
    
    # PREPARE REGISTERS FOR MULTIPLE
        p = expand_dim(p, 'C_reg', LANE, 'itt') #, unsafe_disable_checks=True)
    #if MR != LANE:
        p = expand_dim(p, 'C_reg', MR//LANE, 'it') #, unsafe_disable_checks=True)
    #p = expand_dim(p, 'C_reg', NR, 'j', unsafe_disable_checks=True)
        if NR / LANE > 1:
            p = expand_dim(p, 'C_reg', ENR, 'jt * {} + jtt '.format(LANE)) #, unsafe_disable_checks=True)
            p = lift_alloc(p, 'C_reg', n_lifts=5)
            p = autofission(p, p.find('{}[_] = _'.format('C_reg')).after(), n_lifts=5)
            p = autofission(p, p.find('{}[jtt + {} * jt,itt+{}*it] = _'.format('C',LANE,LANE)).before(), n_lifts=5)
        else:
            p = expand_dim(p, 'C_reg', ENR, 'jtt'.format(LANE)) #, unsafe_disable_checks=True)
            p = lift_alloc(p, 'C_reg', n_lifts=4)
            p = autofission(p, p.find('{}[_] = _'.format('C_reg')).after(), n_lifts=4)
            p = autofission(p, p.find('{}[jtt,itt+{}*it] = _'.format('C',LANE)).before(), n_lifts=4)
        p = simplify(p)
    #MOVE LOADS AND STORES OF THE CASE WHEN MULTIPLE
    
        p = set_memory(p, 'C_reg',intrinsics['mem'])
        p = replace_all(p, intrinsics['load'])
        p = replace_all(p, intrinsics['store'])
        
        edge = True if NR/LANE > 1 else False 
        Buf = 'A'
        Xreg='{}_reg'.format(Buf)
        p = bind_expr(p, '{}[_]'.format(Buf),Xreg)
        p = simplify(p)
        loop = 'i'
        p = expand_dim(p, Xreg , LANE, '{}tt'.format(loop)) #, unsafe_disable_checks=True)
        p = expand_dim(p, Xreg, MR//LANE, '{}t'.format(loop)) #, unsafe_disable_checks=True)
        p = lift_alloc(p, Xreg, n_lifts=5 if edge else 4)
        p = autofission(p, p.find('{}[_] = _'.format(Xreg)).after(),n_lifts=4 if edge else 3)
        p = set_memory(p, 'A_reg', intrinsics['mem'])
        p = replace(p, p.find('for itt in _:_'),intrinsics['load'])
    
        scal = 'B'
        scr = '{}_reg'.format(scal)
        p = bind_expr(p,scal,scr)
        p = expand_dim(p, scr, LANE, 'jtt') #, unsafe_disable_checks=True)
        p = simplify(p)
        if NR // LANE >= 1:
            p = expand_dim(p, scr, NR//LANE, 'jt') #, unsafe_disable_checks=True)
        p = simplify(p)
        p = lift_alloc(p, scr, n_lifts=5 if edge else 4)
        p = autofission(p, p.find('{}[_] = _'.format(scr)).after(), n_lifts=4 if edge else 3)
    
        p = simplify(p)
        p = set_memory(p, 'B_reg', intrinsics['mem'])
        p = replace(p, p.find('for jtt in _:_ #1'),intrinsics['load'])
        p = simplify(p)
        p = unroll_loop(p,'it')
        p = unroll_loop(p,'jtt')
        if edge:
            p = unroll_loop(p,'jt')
        p = unroll_loop(p,'it')
        if edge:
            p = unroll_loop(p,'jt')
        p = reorder_loops(p,'jtt it')
        p = simplify(p)
        p = replace(p, p.find('for itt in _:_'),intrinsics['fmla'])
        p = simplify(p)
        if edge: 
            p = make_tail(p, (MR//LANE)*LANE, (NR//LANE)*LANE, MR, NR, LANE)
        if edge:
            p = moveup(p, "C_regt : _")
            p = reorder_up(p, "for jtt in _:_ #2",n=1)
            p = reorder_up(p, "for jtt in _:_ #1",n=1)
            p = moveup(p, "B_regt : _")
            p = moveup(p, "A_regt : _")
            p = reorder_up(p, "for k in _:_ #1",n=1)
            p = fuse(p,'for k in _:_ #0','for k in _:_ #1')
            p = unroll_loop(p,'it')
            p = unroll_loop(p,'jtt')
            p = reorder_up(p, "for it in _:_ #1",n=1)
            p = reorder_up(p, "for it in _:_ #0",n=1)
            p = unroll_loop(p,'it')
            p = reorder_up(p, "neon_vld_4xf32(_) #{}".format((NR+(MR//LANE)+1+NR//LANE)),n=1)
            p = simplify(p)
        
        p = unroll_loop(p,'jtt')
        p = unroll_loop(p,'it')
        p = simplify(p)
        if edge:
            p = unroll_loop(p,'jt')
            p = reorder_loops(p,'jtt it')
            p = unroll_loop(p,'jtt')
            p = unroll_loop(p,'it')
        p = unroll_loop(p,'it')
        p = unroll_loop(p,'jtt')
        p = simplify(p)
        if edge:
            p = unroll_loop(p,'jt')
            p = unroll_loop(p,'it')
            p = unroll_loop(p,'jtt')
    #p = unroll_loop(p,'it')
    #p = unroll_loop(p,'it')
   # p = unroll_loop(p,'it')
   # p = unroll_loop(p,'it')

    
    #if MR % LANE != 0 or NR % LANE != 0 :
    #    p = reorder_up(p, "for jtt in _:_ #1",n=1)
        #p = reorder_up(p, "for j in _:_ #3",n=1)
        #p = reorder_up(p, "for j in _:_ #1",n=2)
        #p = fuse(p,'for j in _:_ #0','for j in _:_ #1')
    #p = reorder_stmts(p, "for j in _:_ #3\nfor j in _:_ #4")
    
        if edge == False: 
            p = moveup(p, "B_reg : _")
            p = moveup(p, "A_reg : _")
    
        #p = reorder_up(p, "for j in _:_ #4",n=2)
        #p = reorder_up(p, "for j in _:_ #3",n=1)
        #p = reorder_up(p, "for j in _:_ #3",n=1)
        #p = reuse_buffer(p, 'B_reg','B_regt')
    #else:
        else:
            p = reuse_buffer(p, 'A_reg','A_regt')
            p = moveup(p, "B_regt : _")
            p = moveup(p, "B_reg : _")
            #p = moveup(p, "A_regt : _")
            p = moveup(p, "A_reg : _")
        #p = fuse(p,'for j in _:_ #6','for j in _:_ #7')
    
    #while True:
    #    try:
    #        p = unroll_loop(p, "jtt")
    #    except:
    #        break;
    #while True:
    #    try:
    #        p = unroll_loop(p, "jt")
    #    except:
    #        break;
        p = simplify(p)
        p=unrollbuffers(p, "C_reg")
        for i in range(NR):
            try:
                p=unrollbuffers(p, "C_reg_{}".format(i))
            except:
                break;
        p=unrollbuffers(p, "A_reg")
        if NR / LANE > 1:
            try:
                p=unrollbuffers(p, "B_reg")
            except:
                print("Check {}x{}".format(MR,NR))
        #if MR % LANE != 0 or NR % LANE !=0:
        if edge:
            p=unrollbuffers(p, "C_regt")
            for i in range(NR):
                try:
                    p=unrollbuffers(p, "C_regt_{}".format(i))
                except:
                    continue;
            #p=unrollbuffers(p, "A_regt")
            if NR / LANE > 1:
                try:
                    p=unrollbuffers(p, "B_reg")
                except:
                    print("Check {}x{}".format(MR,NR))
    
        p = unroll_loop(p,'i')
        p = unroll_loop(p,'j')
        #print("FINAL",p)
        return p

def howmanyregs(M,N, lane):
    reg_a = M//lane if M % lane == 0 else M//lane + 1
    reg_b = N//lane if N % lane == 0 else N//lane+1
    reg_c = N *  (M//lane if M % lane == 0 else M//lane + 1)
    return reg_a + reg_b + reg_c

m, n, lane, regs  =  (int(x) for x in input().split())
mr = emr =m
nr = enr = n
LANE = lane
if LANE == 4:
    prec="fp32"
elif LANE == 8:
    prec="fp16"
else:
    print("ERROR, this generator is only suitable for floating point dataypes")
    exit()
if (mr == 24 and nr == 24 and LANE == 4) or (mr == 48 and nr == 48 and LANE == 8):
    regs = 1000000
if howmanyregs(emr, enr, LANE) <= regs:

    for i in range(1,mr+1,1):
        for j in range(1,nr+1,1):
            emr = mr = i
            enr = nr = j
            if mr % LANE != 0:
                emr = math.ceil(mr/LANE)*LANE
            if nr % LANE != 0:
                enr = math.ceil(nr/LANE)*LANE
            print("GENERATING {}x{} with {} registers".format(i,j, howmanyregs(emr, enr, LANE)))
            locals()['uk_{0}x{1}_b{2}'.format(i,j,False)] = generator(MR=i, NR=j, pre=prec, EMR = emr, ENR=enr, beta0 = False)
            locals()['uk_{0}x{1}_b{2}'.format(i,j,True)] = generator(MR=i, NR=j, pre=prec,EMR = emr, ENR=enr,beta0 = True)

