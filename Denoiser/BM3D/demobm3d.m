function  demobm3d(  )
y = im2double(imread('barbara.png')); 
sigma=25;
 z = y + sigma/255*randn(size(y));%��������������255��ԭ���ǽ�����������0��1�ڣ������ź�Ҳ���������Χ�ڵ�
 profile='lc';
 print_to_screen=1;
 %y = denoising_dwt(z);
 n=mad(z)%��Ҫ�Ľ���ֻ�ܹ������ͼ��
 [PSNR, y_est] = BM3D(y, z, sigma, profile, print_to_screen);