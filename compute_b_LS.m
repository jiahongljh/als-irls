function [b_LS, e_all, n_removed] = compute_b_LS(F, H, G, Q0, R0, z, N, outlier_thr)
% Empirical autocovariance vector  b_LS  from tau=size(z,2) innovations.
%
% outlier_thr > 0  (ALS-IRLS, recommended=4.5):
%   Detects outlier innovations |e_k| > outlier_thr * sigma_MAD
%   sigma_MAD = 1.4826*MAD(|e|) — robust scale estimate (50% breakdown).
%   Pairs containing an outlier are EXCLUDED from Ce,j computation.
%   With out_mag=15: |e_out|≈26, sigma_MAD≈3.2, threshold≈14.4 → detection≈100%
%
% outlier_thr <= 0 / nargin<8 (ALS baseline):
%   Standard computation using ALL innovations → b_LS contaminated by outliers.

nx=size(F,1); nz=size(H,1); [~,g]=size(G);
tau=size(z,2);
[Pe,~,~]=dare(F',H',G*Q0*G',R0);
K=Pe*H'/(H*Pe*H'+R0);
xp=zeros(nx,1); e_all=zeros(nz,tau);
for k=1:tau
    e_all(:,k)=z(:,k)-H*xp;
    xp=F*(xp+K*e_all(:,k));
end

%% Innovation-level outlier masking
mask=true(1,tau); n_removed=0;
if nargin>=8 && outlier_thr>0
    sigma_e=1.4826*median(abs(e_all(:)));
    sigma_e=max(sigma_e,1e-10);
    for i=1:nz
        mask=mask & (abs(e_all(i,:))<=outlier_thr*sigma_e);
    end
    n_removed=sum(~mask);
end

%% Autocovariance from non-outlier pairs
b_LS=zeros(N*nz^2,1); idx=1;
for lag=0:N-1
    Ce=zeros(nz,nz); cnt=0;
    for k=1:tau-lag
        if mask(k) && mask(k+lag)
            Ce=Ce+e_all(:,k+lag)*e_all(:,k)'; cnt=cnt+1;
        end
    end
    if cnt>0, Ce=Ce/cnt; end
    b_LS(idx:idx+nz^2-1)=Ce(:); idx=idx+nz^2;
end
end
