from __future__ import annotations
from exo import *
from exo import proc
from exo.platforms.neon import *
from exo.stdlib.scheduling import *
import math
@instr("vst1q_s32(&{dst_data}, {src_data});")
def neon_vst_4xi32(dst: [i32][4] @ DRAM, src: [i32][4] @ Neon):
    assert stride(src, 0) == 1
    assert stride(dst, 0) == 1
    
    for i in seq(0, 4):
        dst[i] = src[i]

@instr("{dst_data} = vld1q_s32(&{src_data});")
def neon_vld_4xi32(dst: [i32][4] @ Neon, src: [i32][4] @ DRAM):
    assert stride(src, 0) == 1
    assert stride(dst, 0) == 1
    
    for i in seq(0, 4):
        dst[i] = src[i]

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

@instr("{dst_data} = vmovq_n_f16((float16_t *)0.0);")
def neon_zero_8xf16_new(dst: [f16][8] @ Neon):
    assert stride(dst, 0) == 1
    
    for i in seq(0, 8):
        dst[i] = 0.0

    

def generator(MR,NR, pre,EMR, ENR, beta0, LANE=4, LANET=8, RMR=8, RNR=12):
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
            alpha: i32[1],
            A: i8[KC, MR] @ DRAM,
            B: i8[KC, NR] @ DRAM,
            beta: i32[1],
            C: i32[NR, MR] @ DRAM,
            ):

        assert stride(A, 0) == RMR
        assert stride(A, 1) == 1
        assert stride(B, 0) == RNR
        assert stride(B, 1) == 1
        assert stride(C, 1) == 1

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
            alpha: i32[1],
            A: i8[KC, EMR] @ DRAM,
            B: i8[KC, NR] @ DRAM,
            b: i32[1],
            Ci: i32[NR, EMR] @ DRAM,
            ):
        assert NR <= ENR
        assert MR <= EMR
        assert stride(A, 0) == EMR
        assert stride(A, 1) == 1
        assert stride(B, 0) == ENR
        assert stride(B, 1) == 1
        assert stride(Ci, 1) == 1
        C : i32[ENR,EMR] @ DRAM
        
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
            alpha: i32[1],
            A: i8[KC, EMR] @ DRAM,
            B: i8[KC, ENR] @ DRAM,
            b: i32[1],
            Ci: i32[NR, MR] @ DRAM,
            ):
        assert NR <= ENR
        assert MR <= EMR
        assert stride(A, 0) == EMR
        assert stride(A, 1) == 1
        assert stride(B, 0) == ENR
        assert stride(B, 1) == 1
        assert stride(Ci, 1) == 1
        C : i32[ENR,EMR] @ DRAM
        
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
            pass #print("jtt?")

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
            pass #print("There is not any i loop")
        try:
            t = unroll_loop(t, "j")
        except:
            pass #print("There is not any j loop")
        t=unrollbuffers(t, "C_reg")
    
        return t
    
    def make_lanes(p):
        loop='i'
        p =  divide_loop(p,loop, LANE, ['{}t'.format(loop),'{}tt'.format(loop)], perfect=True)
        loop='j'
        p =  divide_loop(p,loop, LANE, ['{}t'.format(loop),'{}tt'.format(loop)], perfect=True)
        p = simplify(p)
        edge = MR % LANE != 0 or NR % LANE !=0
        p = stage_mem(p, "C[_] += _", "C[jtt + {} * jt, itt + {} * it]".format(LANE,LANE), "C_reg", init_zero=True if beta0 == True or edge == True else False) #REGISTERS FOR MULTIPLE OF LANE
        # PREPARE REGISTERS FOR MULTIPLE
        p = expand_dim(p, 'C_reg', LANE, 'itt') #, unsafe_disable_checks=True)
        p = expand_dim(p, 'C_reg', EMR//LANE, 'it') #, unsafe_disable_checks=True)
        p = expand_dim(p, 'C_reg', ENR, 'jt * {} + jtt'.format(LANE)) #, unsafe_disable_checks=True)
        p = lift_alloc(p, 'C_reg', n_lifts=5)
        p = simplify(p)
        #MOVE LOADS AND STORES OF THE CASE WHEN MULTIPLE
        p = autofission(p, p.find('{}[_] = _'.format('C_reg')).after(), n_lifts=5)
        pat = '{}[jtt + {} * jt,itt+{}*it] = _'.format('C',LANE,LANE) 
        p = autofission(p, p.find(pat).before(), n_lifts=5)
        p = simplify(p)
        
        p = set_memory(p, 'C_reg',Neon)
        p = replace_all(p, neon_zero_4xi32 if beta0 or MR%LANE !=0 or NR%LANE !=0 else  neon_vld_4xi32)
        p = replace_all(p, neon_vst_4xi32)
        
        Buf = 'A'
        Xreg='{}_reg'.format(Buf)
        Xt='{}_temp'.format(Buf)
        p = bind_expr(p, '{}[_]'.format(Buf),Xt)
        p = bind_expr(p, Xt,Xreg)
        p = set_precision(p,'A_reg',"i16") 
        p = set_precision(p,'A_temp',"i8") 
        p = simplify(p)
        loop = 'i'
        mm=math.ceil(EMR/LANET)*LANET
        p = expand_dim(p, Xreg , EMR, 'it * 4 + itt') #, unsafe_disable_checks=True)
        p = expand_dim(p, Xt , mm, 'it * 4 + itt') #, unsafe_disable_checks=True)
        
        p = lift_alloc(p, Xreg, n_lifts=5)
        p = lift_alloc(p, Xt, n_lifts=5)
        p = autofission(p, p.find('{}[_] = _'.format(Xreg)).after(),n_lifts=4)
        p = autofission(p, p.find('{}[_] = _'.format(Xt)).after(),n_lifts=2)
        p = mult_loops(p,p.find('for it in _:_ #1'),'load8')
        p = simplify(p)
        p = set_memory(p, 'A_reg', Neon)
        p = set_memory(p, 'A_temp', Neon)
        if EMR % LANET == 0:
            loop='load8'
            p =  divide_loop(p,loop, LANET, ['{}t'.format(loop),'{}tt'.format(loop)], perfect=True)
            p = divide_dim(p,'A_temp',0,8)
            p = simplify(p)
            p = replace(p,p.find('for load8tt in _:_'),neon_vld_8xi8)
            p = unroll_loop(p,'load8t')
        else:
            if EMR == LANE:
                loop='load8'
                p = divide_dim(p,'A_temp',0,8)
                p = simplify(p)
                p = replace(p,p.find('for load8 in _:_'),neon_vld_8xi8)
            else:
                p = divide_dim(p,'A_temp',0,8)
                p = simplify(p)
                loop='load8'
                p =  divide_loop(p,loop, LANET, ['{}t'.format(loop),'{}tt'.format(loop)], tail='cut')
                p = simplify(p)
                p = replace_all(p,neon_vld_8xi8)
                p = unroll_loop(p,'load8t')
        
        p = mult_loops(p,p.find('for it in _:_ #1'),'l1')
        p = simplify(p)
        loop='l1'
        p =  divide_loop(p,loop, LANET, ['{}t'.format(loop),'{}tt'.format(loop)], tail='cut')
        p = simplify(p)
        loop='l1tt'
        p =  divide_loop(p,loop, LANE, ['{}t'.format(loop),'{}tt'.format(loop)], perfect=True)
        p = simplify(p)
        p = divide_dim(p,'A_reg',0,4)
        p = simplify(p)
        p = unroll_loop(p,'l1ttt')
        if EMR % LANET == 0:
            p = replace(p,p.find('for l1tttt in _:_ '), neon_get_low_8xi16)
            p = replace(p,p.find('for l1tttt in _:_'), neon_get_high_8xi16)
        else:
            if EMR == LANE: 
                p = replace(p,p.find('for l1tttt in _:_ '), neon_get_low_8xi16)
            else:
                p = replace(p,p.find('for l1tttt in _:_ '), neon_get_low_8xi16)
                p = replace(p,p.find('for l1tttt in _:_'), neon_get_high_8xi16)
                p = replace(p,p.find('for l1tt in _:_ '), neon_get_low_8xi16)
        p = simplify(p)
        try:
            p = unroll_loop(p,'l1t')
        except:
            pass
        
        scal = 'B'
        scr = '{}_reg'.format(scal)
        scrt = '{}_temp'.format(scal)
        p = bind_expr(p,scal,scrt)
        p = bind_expr(p,scrt,scr)
        p = set_precision(p,'B_reg',"i16") 
        p = set_precision(p,'B_temp',"i8") 
        p = set_memory(p, 'B_reg', Neon)
        p = set_memory(p, 'B_temp', Neon)
        
        mm=math.ceil(ENR/LANET)*LANET
        p = expand_dim(p, scr, ENR, 'jt * 4 + jtt') #, unsafe_disable_checks=True)
        p = expand_dim(p, scrt, mm, 'jt * 4 + jtt') #, unsafe_disable_checks=True)
        p = lift_alloc(p, scr, n_lifts=5)
        p = lift_alloc(p, scrt, n_lifts=5)
        p = autofission(p, p.find('{}[_] = _'.format(scr)).after(),n_lifts=4)
        p = autofission(p, p.find('{}[_] = _'.format(scrt)).after(),n_lifts=2)
        p = mult_loops(p,p.find('for jt in _:_ #1'),'load8')
        p = simplify(p)
        
        if ENR % LANET == 0:
            loop='load8'
            p =  divide_loop(p,loop, LANET, ['{}t'.format(loop),'{}tt'.format(loop)], perfect=True)
            p = divide_dim(p,'B_temp',0,8)
            p = simplify(p)
            p = replace(p,p.find('for load8tt in _:_'),neon_vld_8xi8)
            p = unroll_loop(p,'load8t')
        else:
            if ENR == LANE:
                loop='load8'
                p = divide_dim(p,'B_temp',0,8)
                p = simplify(p)
                p = replace(p,p.find('for load8 in _:_'),neon_vld_8xi8)
            else:
                p = divide_dim(p,'B_temp',0,8)
                p = simplify(p)
                loop='load8'
                p =  divide_loop(p,loop, LANET, ['{}t'.format(loop),'{}tt'.format(loop)], tail='cut')
                p = simplify(p)
                p = replace_all(p,neon_vld_8xi8)
                p = unroll_loop(p,'load8t')
        
        p = mult_loops(p,p.find('for jt in _:_ #1'),'l1')
        p = simplify(p)
        loop='l1'
        p =  divide_loop(p,loop, LANET, ['{}t'.format(loop),'{}tt'.format(loop)], tail='cut')
        p = simplify(p)
        loop='l1tt'
        p =  divide_loop(p,loop, LANE, ['{}t'.format(loop),'{}tt'.format(loop)], perfect=True)
        p = simplify(p)
        p = divide_dim(p,'B_reg',0,4)
        p = simplify(p)
        p = unroll_loop(p,'l1ttt')
        if ENR % LANET == 0:
            p = replace(p,p.find('for l1tttt in _:_ '), neon_get_low_8xi16)
            p = replace(p,p.find('for l1tttt in _:_'), neon_get_high_8xi16)
        else:
            if ENR == LANE: 
                p = replace(p,p.find('for l1tttt in _:_ '), neon_get_low_8xi16)
            else:
                p = replace(p,p.find('for l1tttt in _:_ '), neon_get_low_8xi16)
                p = replace(p,p.find('for l1tttt in _:_'), neon_get_high_8xi16)
                p = replace(p,p.find('for l1tt in _:_ '), neon_get_low_8xi16)
        p = simplify(p)
        try:
            p = unroll_loop(p,'l1t')
        except:
            pass
        p = simplify(p)
        
        p = unroll_loop(p,'it')        
        p = unroll_loop(p,'jtt')        
        p = unroll_loop(p,'jt') 

        p = reorder_loops(p,'jtt it')
        p = replace_all(p, neon_vmlal_8xi16_8xi16)

        p = simplify(p)
        try:
            p = unroll_loop(p,'lt')
        except:
            pass
        try:
            p = unroll_loop(p,'l2t')
        except:
            pass
        p = unroll_loop(p,'jtt')
        p = unroll_loop(p,'it')
        p = unroll_loop(p,'jt')
        p = unroll_loop(p,'it')
        p = unroll_loop(p,'jtt')
        p = unroll_loop(p,'jt')
        p = simplify(p)
        try:
            p=unrollbuffers(p, "A_reg")
        except:
            pass #print("A buffer does not unroll")
        try:
            p=unrollbuffers(p, "A_temp")
        except:
            pass #print("A temp buffer does not unroll")
        
        try:
            p=unrollbuffers(p, "B_reg")
        except:
            pass #print("B buffer does not unroll")
        try:
            p=unrollbuffers(p, "B_temp")
        except:
            pass #print("B temp buffer does not unroll")
        
        p=unrollbuffers(p, "C_reg")
        for i in range(ENR if NR % LANE == 0 else ENR):
            try:
                p=unrollbuffers(p, "C_reg_{}".format(i))
            except:
                break;
        p =  divide_loop(p,'k', 4, ['kt','ktt'], tail='cut')
        p = unroll_loop(p,'ktt') 
        return p
    
    if pre == "fp32":
        precision="f32"
        LANE=4
        intrinsics = {'load': neon_vld_4xf32, 'store': neon_vst_4xf32, 'fmla':  neon_vfmla_4xf32_4xf32,
            'bcast': neon_broadcast_4xf32, 'vmul':neon_vmul_4xf32, 'zeros': neon_zero_4xf32, 'mem':Neon}
    
    elif pre == "fp16":
        precision="f16"
        LANE=8
        intrinsics = {'load': neon_vld_8xf16, 'store': neon_vst_8xf16, 'fmla':  neon_vfmla_8xf16_8xf16,
            'bcast': neon_broadcast_8xf16, 'vmul':neon_vmul_8xf16, 'zeros': neon_zero_8xf16_new, 'mem':Neon}
    else:
        pass #print("New precission")
    
    
    #if MR % LANE == 0 and NR % LANE == 0:
    if True:
        if MR % LANE != 0 or NR % LANE != 0:
            p=ukernel_edge_esp
            p = set_window(p, "Ci", True)
        else:
            p=ukernel_main
            p = set_window(p, "C", True)
        p = set_window(p, "A", True)
        p = set_window(p, "B", True)
        p = rename(p, "gemm_{}_{}x{}_b{}_{}_{}".format("NEON", MR,NR,0 if beta0 else 1, "col",pre))
        p = p.partial_eval(MR=MR,NR=NR)
        p = simplify(p)
        p = make_lanes(p)
        p = simplify(p)
        if MR % LANE != 0 or NR % LANE != 0:
            p = unroll_loop(p,'i')
            p = unroll_loop(p,'j')
        return p


def howmanyregs(M,N, lane):
    reg_a = M//lane if M % lane == 0 else M//lane + 1
    reg_b = N//lane if N % lane == 0 else N//lane+1
    reg_c = N *  (M//lane if M % lane == 0 else M//lane + 1)
    return reg_a + reg_b + reg_c

m, n, lane =  (int(x) for x in input().split())
mr = emr = m
nr = enr = n
LANE = lane
beta=0.0
prec="i32"
maxi = maxj = 0
LANET = 8
if howmanyregs(m,n,lane) <= 36:
    for i in range(1,m+1,1):
        for j in range(1,n+1,1):
                emr = mr = i
                enr = nr = j
                if mr % LANE != 0:
                    emr = math.ceil(mr/m)*m
                if nr % LANE != 0:
                    enr = math.ceil(nr/n)*n
                print("GENERATING {}x{} with {} registers".format(i,j, howmanyregs(emr, enr, LANE)))
                locals()['uk_{0}x{1}_b{2}'.format(i,j,False)] = generator(MR=i, NR=j, pre=prec, EMR = emr, ENR=enr,   beta0 = False, LANE=4, LANET=8, RMR=m, RNR=n)
                locals()['uk_{0}x{1}_b{2}'.format(i,j,True)] = generator(MR=i, NR=j, pre=prec,EMR = emr, ENR=enr,    beta0 = True, LANE=4, LANET=8, RMR=m, RNR=n)
                if i > maxi:
                    maxi = i
                if j > maxj:
                    maxj=j

#from generate_matrix import generate_file

#generate_file(maxi,maxj,LANE,'NEON',prec)
