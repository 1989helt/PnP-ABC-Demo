function x = PnPABC2D_Deblur_Inpaint(opts, x0, y, ker, blur, M) %% Pixel value ranges in [0,255]
lambda1 = opts.lambda1 ; 
lambda2 = opts.lambda2 ; 
mu1 = opts.mu1   ;  
mu2 = opts.mu2   ;  
mu3 = opts.mu3   ;  
tol = opts.tol  ;   
maxit = opts.maxit  ;  
%%%***************************************************************
[m,n] = size(x0) ; 
cker=rot90(ker,2);
A = @(x) blur(x,ker); AT = @(x) blur(x,cker); % Convolution operator
%%%***************************************************************
eigenP=eigenofP(ker,mu1,mu2,mu3,m,n);
%%%***************************************************************
FT = @(x)fft2(x);
IFT = @(x)ifft2(x);

MTy = M.*y ;

%%%***************************************************************
x = y ; z1 = y; z2 = y; z3 = y; %%% Initilization 
d1 = zeros(m,n) ; d2 = zeros(m,n) ;  d3 = zeros(m,n) ;  
%%%***************************************************************

PSNR_out = [] ;    ERROR_out = [] ; 

%% Main function
for nstep=1:maxit 
%%%%%%%%%%%%%%%%%%%%%%%%%%%    
%%% z1-subproblem 
%%%%%%%%%%%%%%%%%%%%%%%%%%%
Z1 = x - d1 ; Sigma = sqrt(lambda1/mu1) ;
z1 = DenoiserBM3D(Z1,Sigma) ; %%% BM3D denoiser range [0,255]

%%%%%%%%%%%%%%%%%%%%%%%%%%%    
%%% z2-subproblem  
%%%%%%%%%%%%%%%%%%%%%%%%%%%
Z2 = x - d2 ; Sigma = sqrt(lambda2/mu2) ;
z2 = FFDNet(Z2/255,Sigma/255)*255 ; %%% FFDNet denoiser range [0,1]   

%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%%% z3-subproblem (MASK OPERATION)
%%%%%%%%%%%%%%%%%%%%%%%%%%%
z3 = ( mu3*(A(x)-d3) + MTy )./(mu3 + M) ;    

%%%%%%%%%%%%%%%%%%%%%%%%%%%    
%%% x-subproblem
%%%%%%%%%%%%%%%%%%%%%%%%%%%
x_prev = x ;
x = IFT( FT( mu1*(z1+d1) + mu2*(z2+d2) + mu3*AT(z3+d3) )./eigenP );
x(x>255) = 255; x(x<0) = 0;  %% range [0 255]    

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Lagrangian d1; d2; d3 updating
%%%%%%%%%%%%%%%%%%%%%%%%%%%
d1 = d1 + z1 - x ;
d2 = d2 + z2 - x ;
d3 = d3 + z3 - A(x) ;

%%% Check the tolerence      
error=norm(x-x_prev,'fro')/norm(x,'fro'); 
disp(['error on step ' num2str(nstep)  ' is ' num2str(error) ', '...
    'and PSNR is ' num2str(psnr(x/255,x0/255)) ', and SSIM is ' num2str(ssim_index(x,x0)) ] );  

psnr_out = psnr(x/255,x0/255) ;  error_out = error ;
PSNR_out(nstep) = psnr_out ;     ERROR_out(nstep) = error_out ;

 if error<tol
        break;
 end
end

figure(1999) ; grid on ;
subplot(1,3,1); plot(1:length(ERROR_out), ERROR_out) ;  grid on ;
subplot(1,3,2); plot(1:length(PSNR_out), PSNR_out) ;  grid on ;
 
function eigenP=eigenofP(ker,mu1,mu2,mu3,m,n)
[nker,mker]=size(ker);
tmp=zeros(m,n);tmp(1:nker,1:mker)=ker;
tmp=circshift(tmp,[-floor(nker/2),-floor(mker/2)]);
eigenP = mu3*( abs(fft2(tmp)).^2 ) + mu1 + mu2;   
