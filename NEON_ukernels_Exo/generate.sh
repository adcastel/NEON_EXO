#/bin/bash
for ARCH in NEON
do
    for prec in fp16
    do
	if [ $prec = fp32 ]; then
	    ini=4
	    end=24
	    step=4
	    lane=4
	else
	    ini=8
	    end=8
	    step=8
	    lane=8
	fi
        for mr in $(seq ${ini} ${step} ${end}); 
        do 
	    for nr in $(seq ${ini} ${step} ${end}); 
	    do  
		ff=kernels_${ARCH}_${mr}x${nr}_${prec}
   	        #if ! test -f kernels/${ARCH}/${mr}x${nr}/${ff}.c ; then
		    echo "${mr} ${nr} ${lane} 36 | exocc -o kernels/${ARCH}/${mr}x${nr} --stem ${ff} NEON_generator_fp.py;"
		    echo "${mr} ${nr} ${lane} 36" | exocc -o kernels/${ARCH}/${mr}x${nr} --stem ${ff} NEON_generator_fp.py;
   	            if test -f kernels/${ARCH}/${mr}x${nr}/${ff}.c ; then
		        echo "python3 exo_to_opt_converter.py kernels/${ARCH}/${mr}x${nr}/${ff}.c kernels/${ARCH}/${mr}x${nr}/${ff}.c 1 ${mr} ${nr} ${prec} ${ARCH}" 
		        python3 exo_to_opt_converter.py kernels/${ARCH}/${mr}x${nr}/${ff}.c kernels/${ARCH}/${mr}x${nr}/${ff}.c 1 ${mr} ${nr} ${prec} ${ARCH} 
		        echo "python3 exo_to_opt_converter.py kernels/${ARCH}/${mr}x${nr}/${ff}.h kernels/${ARCH}/${mr}x${nr}/${ff}.h 1 ${mr} ${nr} ${prec} ${ARCH}" 
		        python3 exo_to_opt_converter.py kernels/${ARCH}/${mr}x${nr}/${ff}.h kernels/${ARCH}/${mr}x${nr}/${ff}.h 1 ${mr} ${nr} ${prec} ${ARCH} 
		        echo "python3 generate_matrix.py ${mr} ${nr} ${lane} ${ARCH} ${prec} ${prec} ${prec}"
		        python3 generate_matrix.py ${mr} ${nr} ${lane} ${ARCH} ${prec} ${prec} ${prec}
		        echo "make MR=${mr} NR=${nr} DTYPE=${prec} ARCH=NEON"
		        make MR=${mr} NR=${nr} DTYPE=${prec} ARCH=NEON
		        echo "./test_uk 1 ${mr} 1 ${nr} 512 100000"
		        ./test_uk 1 ${mr} 1 ${nr} 512 100000 > ${ARCH}_${mr}x${nr}_${prec}.dat
		   else
			   echo "${mr}x${nr} has not been build"
		   fi
                #fi
    	   done; 
        done; 
    done; 
done; 
