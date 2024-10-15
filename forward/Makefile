forward-lib.so: forward.cu
	nvcc $^ -O3 -lineinfo -arch=native --shared -Xcompiler -fPIC -o $@

clean:
	rm -f forward-lib.so
