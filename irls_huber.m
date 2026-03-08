function [theta, w_out, iter_used] = irls_huber(A, b, delta, T, xi, theta0)
% IRLS with Huber weights (Algorithm 1, OR_ALS paper).
% Called on CLEAN b_LS (after innovation-level outlier removal):
%   provides additional robustness against residual contamination.
% delta<=0 → adaptive: 1.5*MAD(b - A*theta0)
m=size(A,1);
if delta<=0
    r0=b-A*theta0;
    delta=1.5*median(abs(r0-median(r0)));
    delta=max(delta,1e-10);
end
theta=theta0; w_out=ones(m,1);
for iter_used=1:T
    r=b-A*theta;
    w=ones(m,1); big=abs(r)>delta; w(big)=delta./abs(r(big));
    AW=A'*diag(w);
    theta_new=(AW*A)\(AW*b);
    if norm(theta_new-theta)<xi, theta=theta_new; w_out=w; return; end
    theta=theta_new; w_out=w;
end
end
