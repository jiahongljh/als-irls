function [x_est, P_est, rmse] = student_t_kf(F, H, GQG, R, z, x_true, nu)
% Student-t VB KF (Huang et al., IEEE TAES 53(3):1545-1556, 2017).
nx=size(F,1); nz=size(H,1); T=size(z,2);
x=zeros(nx,1); P=eye(nx)*10;
x_est=zeros(nx,T); P_est=zeros(nx,nx,T);
for k=1:T
    xp=F*x; Pp=F*P*F'+GQG; e0=z(:,k)-H*xp;
    u=1;
    for it=1:5
        S=H*Pp*H'+R/u; u_new=(nu+nz)/(nu+e0'/S*e0);
        if abs(u_new-u)<1e-5, u=u_new; break; end; u=u_new;
    end
    K=Pp*H'/(H*Pp*H'+R/u); x=xp+K*e0;
    P=(eye(nx)-K*H)*Pp*(eye(nx)-K*H)'+K*(R/u)*K'; P=(P+P')/2;
    x_est(:,k)=x; P_est(:,:,k)=P;
end
rmse=[];
if ~isempty(x_true)
    err=x_true(:,1:T)-x_est; rmse=sqrt(mean(sum(err.^2,1)));
end
end
