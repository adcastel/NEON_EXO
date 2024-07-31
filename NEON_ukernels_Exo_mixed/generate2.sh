#/bin/bash
for m in 8 4 12 16 20 24
do
	for n in 12 4 8 16 20 24
	do
    echo "${m} ${n} 4" | exocc -o test_esp_${m}_${n} --stem kernel_col NEON_generator.py & 
	done
done
