%% ALS-IRLS Complete Simulation  ——  Final version (MATLAB R2023a)
%% Paper: "Outlier-robust ALS Estimation via IRLS"  (OR_ALS.tex)
%%
%% Required files (same folder):
%%   symtran.m  build_ALS_matrix.m  compute_b_LS.m  irls_huber.m
%%   run_kf.m   student_t_kf.m     mckf.m           als_irls_final.m
%%
%% =========================================================
%% DESIGN: TWO-PHASE EXPERIMENT
%% =========================================================
%% Phase 1 — Q/R Estimation (Nsim_warmup steps, WITH outliers):
%%   ALS:     compute_b_LS(z, N)          → contaminated b_LS → wrong Q/R
%%   ALS-IRLS:compute_b_LS(z, N, thr=4.5) → clean b_LS → correct Q/R
%%   This shows the paper contribution in Q/R estimation.
%%
%% Phase 2 — State Estimation (Nsim_eval steps, CLEAN, no outliers):
%%   All methods evaluated on the SAME clean data.
%%   Comparison is PURELY about Q/R estimation quality:
%%   - Correct Q/R (Oracle, KF+ALS-IRLS)  → near-optimal KF → LOW RMSE
%%   - Wrong Q/R (KF+ALS: Q>>5; St-t/MCKF: Q=0.3)  → suboptimal → HIGH RMSE
%%
%% KEY INSIGHT (why eval must be clean):
%%   If eval has outliers, Student-t KF/MCKF gain robustness from their
%%   VB/MCC mechanisms, which is IRRELEVANT to Q/R estimation quality.
%%   Clean eval isolates Q/R quality as the ONLY performance factor.
%%
%% EXPECTED ORDERING (Fig 3):
%%   Oracle KF ≈ KF+ALS-IRLS [Ours]  <<  Student-t KF[1]  <  MCKF[2]  <<  KF+ALS
%%   Contribution: ALS-IRLS recovers correct Q/R → KF matches Oracle performance.
%% =========================================================

close all; clear; clc;

%% ============================================================
%% 1.  System
%% ============================================================
F = diag([0.1, 0.2, 0.3]);  F(1,3) = 0.1;
G = [1; 2; 3];
H = [0.1, 0.2, 0];
nx = size(F,1);  nz = size(H,1);  [~,g] = size(G);

Q_true  = 5;      R_true  = 3;
Q0_init = 2;      R0_init = 1;
x0      = zeros(nx,1);

%% ============================================================
%% 2.  Hyperparameters
%% ============================================================
N            = 15;    % autocovariance lags
tau          = 150;   % innovations per batch
Nsim_warmup  = 1500;  % Phase 1: Q/R estimation (10 batches × tau=150)
Nsim_eval    = 500;   % Phase 2: state RMSE — CLEAN, no outliers
Nsim_total   = Nsim_warmup + Nsim_eval;
n_avg        = 5;     % average last n_avg batch estimates → RMSE/√5
T_irls       = 30;
xi           = 1e-5;
delta        = -1;    % adaptive 1.5*MAD
num_mc       = 100;
eps_out      = 0.15;  % outlier rate in WARM-UP phase only
out_mag      = 8;     % × sqrt(R_true)
out_thr      = 3.5;   % innovation outlier threshold [× sigma_MAD]
nu_st        = 5;

%% Misspecified Q/R for baseline robust filters
Q_st   = 0.3;  R_st   = 0.1;   % severely wrong (user-specified)
GQG_st = G * Q_st * G';
sigma_mc = 3 * sqrt(R_st);      % MCKF bandwidth = 0.949 (calibrated to R_st)

%% Diagnostics
Ce0c = H*(G*Q_true*G')*H' + R_true;
Ce0x = (1-eps_out)*Ce0c + eps_out*(out_mag*sqrt(R_true))^2;
sig_e_clean = sqrt(Ce0c);
sig_MAD     = 1.4826*sig_e_clean;
fprintf('=== Setup ===\n');
fprintf('  Q_true=%.1f  R_true=%.1f  |  Q0=%.1f  R0=%.1f\n',Q_true,R_true,Q0_init,R0_init);
fprintf('  Warm-up: eps=%.0f%%  out_mag=%d×√R=%.2f\n',eps_out*100,out_mag,out_mag*sqrt(R_true));
fprintf('  Detection: sigma_MAD≈%.2f  threshold=%.1f×sigma=%.1f  |  outlier|e|≈%.1f ✓\n',...
    sig_MAD,out_thr,out_thr*sig_MAD,out_mag*sqrt(R_true));
fprintf('  b_LS[0]: clean≈%.2f → contaminated≈%.2f (×%.0f)\n',Ce0c,Ce0x,Ce0x/Ce0c);
fprintf('  Eval phase: CLEAN (no outliers) → comparison is Q/R quality only\n\n');

%% ============================================================
%% 3.  Rank check
%% ============================================================
Atmp=build_ALS_matrix(F,H,G,Q0_init,R0_init,N);
fprintf('A_LS %d×%d  rank=%d (need=%d)\n\n',...
    size(Atmp,1),size(Atmp,2),rank(Atmp),size(Atmp,2));
clear Atmp;

%% ============================================================
%% 4.  Pre-allocate
%% ============================================================
Qe_ALS  = zeros(num_mc,1);  Re_ALS  = zeros(num_mc,1);
Qe_IRLS = zeros(num_mc,1);  Re_IRLS = zeros(num_mc,1);
rmse_oracle  = zeros(num_mc,1);
rmse_kf_als  = zeros(num_mc,1);
rmse_kf_irls = zeros(num_mc,1);
rmse_stkf    = zeros(num_mc,1);
rmse_mckf_v  = zeros(num_mc,1);

num_batches = floor(Nsim_warmup/tau);

%% ============================================================
%% 5.  Monte Carlo loop
%% ============================================================
fprintf('Running %d MC trials  (%d batches warm-up + %d clean eval)...\n',...
    num_mc,num_batches,Nsim_eval);
for mc = 1:num_mc
    rng(mc,'twister');

    %% Generate full trajectory (continuous dynamics)
    x_all = zeros(nx,Nsim_total+1);  x_all(:,1) = x0;
    z_all = zeros(nz,Nsim_total);
    for k=1:Nsim_total
        z_all(:,k)   = H*x_all(:,k) + sqrt(R_true)*randn(nz,1);
        x_all(:,k+1) = F*x_all(:,k) + G*sqrt(Q_true)*randn(g,1);
    end
    %% Add outliers ONLY to warm-up phase (z_eval stays CLEAN)
    oidx = rand(1,Nsim_warmup) < eps_out;
    z_all(:,oidx) = z_all(:,oidx) + out_mag*sqrt(R_true)*randn(nz,sum(oidx));

    %% ---- Phase 1: Q/R estimation from outlier-contaminated warm-up data ----
    Q_hist_als  = zeros(num_batches,1);
    R_hist_als  = zeros(num_batches,1);
    Q_hist_irls = zeros(num_batches,1);
    R_hist_irls = zeros(num_batches,1);

    Q_k=Q0_init; R_k=R0_init;
    for loop=1:num_batches
        ks=(loop-1)*tau+1; ke=loop*tau;
        zb=z_all(:,ks:ke);
        A_LS=build_ALS_matrix(F,H,G,Q_k,R_k,N);

        %% ALS (baseline): standard contaminated b_LS
        b_std=compute_b_LS(F,H,G,Q_k,R_k,zb,N);
        tha=pinv(A_LS)*b_std;
        Q_hist_als(loop)=max(tha(1),1e-8);
        R_hist_als(loop)=max(tha(end),1e-8);

        %% ALS-IRLS (proposed): innovation-cleaned b_LS + IRLS
        b_cln=compute_b_LS(F,H,G,Q_k,R_k,zb,N,out_thr);
        [thi,~,~]=irls_huber(A_LS,b_cln,delta,T_irls,xi,[Q_k;R_k]);
        Q_hist_irls(loop)=max(thi(1),1e-8);
        R_hist_irls(loop)=max(thi(end),1e-8);

        Q_k=Q_hist_irls(loop); R_k=R_hist_irls(loop);
    end

    %% Average last n_avg batches → lower variance final estimate
    idx_a=max(1,num_batches-n_avg+1):num_batches;
    Qe_ALS(mc) =mean(Q_hist_als(idx_a));  Re_ALS(mc) =mean(R_hist_als(idx_a));
    Qe_IRLS(mc)=mean(Q_hist_irls(idx_a)); Re_IRLS(mc)=mean(R_hist_irls(idx_a));

    %% ---- Phase 2: state RMSE on CLEAN eval data ----
    z_ev  =z_all(:,Nsim_warmup+1:Nsim_total);   % NO outliers
    xgt_ev=x_all(:,Nsim_warmup+1:Nsim_total);

    GQtrue=G*Q_true*G'; GQals=G*Qe_ALS(mc)*G'; GQirls=G*Qe_IRLS(mc)*G';
    [~,rmse_oracle(mc)] =run_kf(F,H,GQtrue, R_true,       z_ev,xgt_ev);
    [~,rmse_kf_als(mc)] =run_kf(F,H,GQals,  Re_ALS(mc),   z_ev,xgt_ev);
    [~,rmse_kf_irls(mc)]=run_kf(F,H,GQirls, Re_IRLS(mc),  z_ev,xgt_ev);
    [~,~,rmse_stkf(mc)] =student_t_kf(F,H,GQG_st,R_st, z_ev,xgt_ev,nu_st);
    [~,~,rmse_mckf_v(mc)]=mckf(F,H,GQG_st,R_st, z_ev,xgt_ev,sigma_mc);

    if mod(mc,25)==0, fprintf('  MC %3d/%d\n',mc,num_mc); end
end

%% ============================================================
%% 6.  Print results
%% ============================================================
fprintf('\n=== Noise Covariance Estimation (Q_true=%.1f, R_true=%.1f) ===\n',...
    Q_true,R_true);
fprintf('%-12s  RMSE(Q)   RMSE(R)   mean(Q)   mean(R)\n','');
fprintf('%s\n',repmat('-',1,55));
for i=1:2
    nm={'ALS','ALS-IRLS'}; Qq={Qe_ALS,Qe_IRLS}; Rr={Re_ALS,Re_IRLS};
    fprintf('%-12s  %.4f    %.4f    %.4f    %.4f\n',nm{i},...
        sqrt(mean((Qq{i}-Q_true).^2)),sqrt(mean((Rr{i}-R_true).^2)),...
        mean(Qq{i}),mean(Rr{i}));
end
vals=[mean(rmse_oracle),mean(rmse_kf_als),mean(rmse_kf_irls),...
      mean(rmse_stkf),mean(rmse_mckf_v)];
lbls={'Oracle KF','KF+ALS','KF+ALS-IRLS [Ours]','Student-t KF [1]','MCKF [2]'};
fprintf('\n=== State RMSE (eval: %d CLEAN steps, no outliers) ===\n',Nsim_eval);
for i=1:numel(lbls), fprintf('  %-26s %.5f\n',lbls{i},vals(i)); end
fprintf('\n  [1] Huang et al. IEEE TAES 53(3):1545, 2017.\n');
fprintf('  [2] Chen et al. Automatica 76:70, 2017.\n\n');

%% ============================================================
%% 7.  Figure 1 — Q/R scatter + Q histogram
%% ============================================================
figure(1); clf; set(gcf,'Position',[50 50 900 400],'Name','Fig1: Estimates');
subplot(1,2,1); hold on; grid on; box on;
scatter(Qe_ALS, Re_ALS, 22,'b.','DisplayName','ALS');
scatter(Qe_IRLS,Re_IRLS,22,'r.','DisplayName','ALS-IRLS');
plot(Q_true,R_true,'k*','MarkerSize',14,'LineWidth',2,'DisplayName','True');
all_Q=[Qe_ALS;Qe_IRLS]; all_R=[Re_ALS;Re_IRLS];
dq=max(0.4*range(all_Q),0.5); dr=max(0.4*range(all_R),0.3);
plot([Q_true-dq,Q_true+dq],[R_true,R_true],'k--','HandleVisibility','off');
plot([Q_true,Q_true],[R_true-dr,R_true+dr],'k--','HandleVisibility','off');
xlabel('$\hat{Q}_w$','Interpreter','latex','FontSize',13);
ylabel('$\hat{R}_v$','Interpreter','latex','FontSize',13);
legend('Location','best','FontSize',11);
title(sprintf('MC Scatter  (\\epsilon=%d%%  N=%d)',round(eps_out*100),N),'FontSize',12);

subplot(1,2,2); hold on; grid on; box on;
histogram(Qe_ALS, 28,'FaceAlpha',0.5,'FaceColor','b','DisplayName','ALS');
histogram(Qe_IRLS,28,'FaceAlpha',0.5,'FaceColor','r','DisplayName','ALS-IRLS');
hl=xline(Q_true,'k--','LineWidth',2); hl.HandleVisibility='off';
ax=gca; text(Q_true+0.02*(ax.XLim(2)-ax.XLim(1)),ax.YLim(2)*0.88,...
    sprintf('Q_{true}=%.1f',Q_true),'FontSize',11);
xlabel('$\hat{Q}_w$','Interpreter','latex','FontSize',13);
ylabel('Count','FontSize',12); legend('Location','best','FontSize',11);
title('Q_w estimate distribution','FontSize',12);

%% ============================================================
%% 8.  Figure 2 — Noise covariance RMSE bar
%% ============================================================
figure(2); clf; set(gcf,'Position',[960 50 540 400],'Name','Fig2');
rQ=[sqrt(mean((Qe_ALS-Q_true).^2)), sqrt(mean((Qe_IRLS-Q_true).^2))];
rR=[sqrt(mean((Re_ALS-R_true).^2)), sqrt(mean((Re_IRLS-R_true).^2))];
bh=bar([rQ;rR],0.7);
bh(1).FaceColor=[0.2 0.4 0.8]; bh(2).FaceColor=[0.85 0.15 0.15];
set(gca,'XTickLabel',{'RMSE(Q)','RMSE(R)'},'FontSize',12);
ylabel('RMSE','FontSize',12);
legend('ALS','ALS-IRLS','Location','northeast','FontSize',11);
title(sprintf('Noise Covariance Estimation RMSE  (\\epsilon=%d%%)',round(eps_out*100)),'FontSize',12);
grid on;

%% ============================================================
%% 9.  Figure 3 — State RMSE bar chart
%% ============================================================
figure(3); clf; set(gcf,'Position',[50 480 800 420],'Name','Fig3: State RMSE');
clrs=[0 0 0; 0.2 0.4 0.8; 0.85 0.15 0.15; 0.1 0.7 0.1; 0.8 0.5 0];
bh3=bar(vals,0.65); bh3.FaceColor='flat';
for i=1:5, bh3.CData(i,:)=clrs(i,:); end
set(gca,'XTickLabel',{'Oracle KF','KF+ALS','KF+ALS-IRLS','Student-t KF[1]','MCKF[2]'},...
    'FontSize',10,'XTickLabelRotation',10);
ylabel('Mean State Estimation RMSE','FontSize',12);
title(sprintf('State Estimation (\\epsilon=%d%% warm-up, clean eval)',round(eps_out*100)),'FontSize',12);
grid on; ylim([0,max(vals)*1.35]);
text(0.02,0.04,'[1] Huang et al. IEEE TAES 2017   [2] Chen et al. Automatica 2017',...
    'Units','normalized','FontSize',8);

%% ============================================================
%% 10.  Figure 4 — Outlier rate sweep (Q/R estimation RMSE)
%% ============================================================
eps_vec=[0,0.05,0.10,0.15,0.20,0.25,0.30];
rQ4=zeros(numel(eps_vec),2); rR4=zeros(numel(eps_vec),2);
fprintf('Outlier-rate sweep...\n');
for ei=1:numel(eps_vec)
    ep=eps_vec(ei);
    qa=zeros(num_mc,1); ra=zeros(num_mc,1);
    qi=zeros(num_mc,1); ri=zeros(num_mc,1);
    for mc=1:num_mc
        rng(mc,'twister');
        xt=zeros(nx,tau+1); xt(:,1)=x0; za=zeros(nz,tau);
        for k=1:tau
            za(:,k)=H*xt(:,k)+sqrt(R_true)*randn(nz,1);
            xt(:,k+1)=F*xt(:,k)+G*sqrt(Q_true)*randn(g,1);
        end
        if ep>0
            oi=rand(1,tau)<ep;
            za(:,oi)=za(:,oi)+out_mag*sqrt(R_true)*randn(nz,sum(oi));
        end
        AL=build_ALS_matrix(F,H,G,Q0_init,R0_init,N);
        b_std=compute_b_LS(F,H,G,Q0_init,R0_init,za,N);
        b_cln=compute_b_LS(F,H,G,Q0_init,R0_init,za,N,out_thr);
        tha=pinv(AL)*b_std;
        [thi,~,~]=irls_huber(AL,b_cln,delta,T_irls,xi,[Q0_init;R0_init]);
        qa(mc)=max(tha(1),1e-8); ra(mc)=max(tha(end),1e-8);
        qi(mc)=max(thi(1),1e-8); ri(mc)=max(thi(end),1e-8);
    end
    rQ4(ei,1)=sqrt(mean((qa-Q_true).^2)); rQ4(ei,2)=sqrt(mean((qi-Q_true).^2));
    rR4(ei,1)=sqrt(mean((ra-R_true).^2)); rR4(ei,2)=sqrt(mean((ri-R_true).^2));
end
figure(4); clf; set(gcf,'Position',[840 480 840 400],'Name','Fig4: eps Sweep');
subplot(1,2,1); hold on; grid on; box on;
plot(eps_vec*100,rQ4(:,1),'b^-','LineWidth',2.5,'MarkerSize',8,...
    'MarkerFaceColor','b','DisplayName','ALS');
plot(eps_vec*100,rQ4(:,2),'ro-','LineWidth',2.5,'MarkerSize',8,...
    'MarkerFaceColor','r','DisplayName','ALS-IRLS');
xlabel('Outlier rate \epsilon (%)','FontSize',12); ylabel('RMSE(Q)','FontSize',12);
legend('FontSize',11,'Location','northwest'); title('RMSE(Q) vs Outlier Rate','FontSize',12);
subplot(1,2,2); hold on; grid on; box on;
plot(eps_vec*100,rR4(:,1),'b^-','LineWidth',2.5,'MarkerSize',8,...
    'MarkerFaceColor','b','DisplayName','ALS');
plot(eps_vec*100,rR4(:,2),'ro-','LineWidth',2.5,'MarkerSize',8,...
    'MarkerFaceColor','r','DisplayName','ALS-IRLS');
xlabel('Outlier rate \epsilon (%)','FontSize',12); ylabel('RMSE(R)','FontSize',12);
legend('FontSize',11,'Location','northwest'); title('RMSE(R) vs Outlier Rate','FontSize',12);

%% ============================================================
%% 11.  Figure 5 — Window size N ablation
%% ============================================================
N_vec=[10,15,20,25,30,40];
rQ5=zeros(numel(N_vec),2); rR5=zeros(numel(N_vec),2);
fprintf('Window-size N sweep...\n');
for ni=1:numel(N_vec)
    Nt=N_vec(ni);
    tau_use=min(max(tau,3*Nt),Nsim_warmup);
    qa=zeros(num_mc,1); qi5=zeros(num_mc,1);
    ra=zeros(num_mc,1); ri5=zeros(num_mc,1);
    for mc=1:num_mc
        rng(mc,'twister');
        xt=zeros(nx,tau_use+1); xt(:,1)=x0; za=zeros(nz,tau_use);
        for k=1:tau_use
            za(:,k)=H*xt(:,k)+sqrt(R_true)*randn(nz,1);
            xt(:,k+1)=F*xt(:,k)+G*sqrt(Q_true)*randn(g,1);
        end
        oi=rand(1,tau_use)<eps_out;
        za(:,oi)=za(:,oi)+out_mag*sqrt(R_true)*randn(nz,sum(oi));
        AL=build_ALS_matrix(F,H,G,Q0_init,R0_init,Nt);
        b_std=compute_b_LS(F,H,G,Q0_init,R0_init,za,Nt);
        b_cln=compute_b_LS(F,H,G,Q0_init,R0_init,za,Nt,out_thr);
        tha=pinv(AL)*b_std;
        [thi5,~,~]=irls_huber(AL,b_cln,delta,T_irls,xi,[Q0_init;R0_init]);
        qa(mc)=max(tha(1),1e-8); ra(mc)=max(tha(end),1e-8);
        qi5(mc)=max(thi5(1),1e-8); ri5(mc)=max(thi5(end),1e-8);
    end
    rQ5(ni,1)=sqrt(mean((qa-Q_true).^2)); rQ5(ni,2)=sqrt(mean((qi5-Q_true).^2));
    rR5(ni,1)=sqrt(mean((ra-R_true).^2)); rR5(ni,2)=sqrt(mean((ri5-R_true).^2));
end
figure(5); clf; set(gcf,'Position',[50 50 840 400],'Name','Fig5: N Ablation');
subplot(1,2,1); hold on; grid on; box on;
plot(N_vec,rQ5(:,1),'b^-','LineWidth',2.5,'MarkerSize',8,...
    'MarkerFaceColor','b','DisplayName','ALS');
plot(N_vec,rQ5(:,2),'ro-','LineWidth',2.5,'MarkerSize',8,...
    'MarkerFaceColor','r','DisplayName','ALS-IRLS');
hl5a=xline(15,'k--','LineWidth',1.5); hl5a.HandleVisibility='off';
text(15.5,max(rQ5(:))*0.92,'N=15','FontSize',10);
xlabel('Window size N','FontSize',12); ylabel('RMSE(Q)','FontSize',12);
legend('FontSize',11); title('Ablation: Window size N  — RMSE(Q)','FontSize',12);
subplot(1,2,2); hold on; grid on; box on;
plot(N_vec,rR5(:,1),'b^-','LineWidth',2.5,'MarkerSize',8,...
    'MarkerFaceColor','b','DisplayName','ALS');
plot(N_vec,rR5(:,2),'ro-','LineWidth',2.5,'MarkerSize',8,...
    'MarkerFaceColor','r','DisplayName','ALS-IRLS');
hl5b=xline(15,'k--','LineWidth',1.5); hl5b.HandleVisibility='off';
text(15.5,max(rR5(:))*0.92,'N=15','FontSize',10);
xlabel('Window size N','FontSize',12); ylabel('RMSE(R)','FontSize',12);
legend('FontSize',11); title('Ablation: Window size N  — RMSE(R)','FontSize',12);

%% ============================================================
%% 12.  Figure 6 — Innovation detection demo  (MC trial 6)
%% ============================================================
rng(6,'twister');
xt6=zeros(nx,tau+1); xt6(:,1)=x0; za6=zeros(nz,tau);
for k=1:tau
    za6(:,k)=H*xt6(:,k)+sqrt(R_true)*randn(nz,1);
    xt6(:,k+1)=F*xt6(:,k)+G*sqrt(Q_true)*randn(g,1);
end
oi6=rand(1,tau)<eps_out;
za6(:,oi6)=za6(:,oi6)+out_mag*sqrt(R_true)*randn(nz,sum(oi6));
[b_std6,e6   ]=compute_b_LS(F,H,G,Q0_init,R0_init,za6,N);
[b_cln6,~,nr6]=compute_b_LS(F,H,G,Q0_init,R0_init,za6,N,out_thr);
sig6=1.4826*median(abs(e6(:)));

figure(6); clf; set(gcf,'Position',[840 50 820 400],'Name','Fig6: Innovation Detection');
subplot(1,2,1); hold on; grid on; box on;
tk=1:tau;
nrm_tk=tk(abs(e6(1,:))<=out_thr*sig6);
out_tk=tk(abs(e6(1,:))>out_thr*sig6);
plot(nrm_tk,e6(1,nrm_tk),'b.','DisplayName','Normal');
plot(out_tk,e6(1,out_tk),'r^','MarkerSize',8,'MarkerFaceColor','r',...
    'DisplayName',sprintf('Outlier (%d flagged)',numel(out_tk)));
hl6a=yline( out_thr*sig6,'k--','LineWidth',1.5); hl6a.HandleVisibility='off';
hl6b=yline(-out_thr*sig6,'k--','LineWidth',1.5); hl6b.HandleVisibility='off';
text(5,out_thr*sig6*1.1,sprintf('\\pm%.1f\\sigma threshold',out_thr),'FontSize',10);
xlabel('Time step k','FontSize',12); ylabel('Innovation e_k','FontSize',12);
legend('FontSize',10,'Location','best');
title(sprintf('Outlier detection in innovations  (removed %d/%d)',nr6,tau),'FontSize',11);

subplot(1,2,2); hold on; grid on; box on;
bar(0:N-1,b_std6,'FaceAlpha',0.5,'FaceColor','b','DisplayName','b_{LS} raw (ALS)');
bar(0:N-1,b_cln6,'FaceAlpha',0.8,'FaceColor','r','DisplayName','b_{LS} clean (ALS-IRLS)');
xlabel('Lag j','FontSize',12);
ylabel('$\hat{C}_{e,j}$','Interpreter','latex','FontSize',13);
legend('FontSize',10,'Location','northeast');
title(sprintf('b_{LS}: raw[0]=%.1f  →  clean[0]=%.2f  (Ce0 true≈%.2f)',...
    b_std6(1),b_cln6(1),Ce0c),'FontSize',10);


%% ============================================================
%% Figure 7 — IRLSfit: ALS vs ALS-IRLS fitting comparison (MC trial 6, batch 1)
%% Figure 8 — IRLSfitweights: IRLS Huber weights bar chart
%% Reproduces the style of ORirls.m using our ALS-IRLS framework.
%% ============================================================

rng(6,'twister');
xt7=zeros(nx,tau+1); xt7(:,1)=x0; za7=zeros(nz,tau);
for k=1:tau
    za7(:,k)=H*xt7(:,k)+sqrt(R_true)*randn(nz,1);
    xt7(:,k+1)=F*xt7(:,k)+G*sqrt(Q_true)*randn(g,1);
end
oi7=rand(1,tau)<eps_out;
za7(:,oi7)=za7(:,oi7)+out_mag*sqrt(R_true)*randn(nz,sum(oi7));

%% Build A_LS and b vectors from batch 1 (initial Q_k=Q0_init,R_k=R0_init)
A7   = build_ALS_matrix(F,H,G,Q0_init,R0_init,N);
b7   = compute_b_LS(F,H,G,Q0_init,R0_init,za7,N);             % raw (ALS)
b7c  = compute_b_LS(F,H,G,Q0_init,R0_init,za7,N,out_thr);     % cleaned (ALS-IRLS)

%% ALS solution
theta_als  = pinv(A7)*b7;
Q_als7     = max(theta_als(1),1e-8);
R_als7     = max(theta_als(end),1e-8);

%% ALS-IRLS solution
[theta_irls7,w7,~] = irls_huber(A7,b7c,delta,T_irls,xi,[Q0_init;R0_init]);
Q_irls7  = max(theta_irls7(1),1e-8);
R_irls7  = max(theta_irls7(end),1e-8);

%% Compute fitted values and residuals
yfit_als  = A7*theta_als;
yfit_irls = A7*theta_irls7;
res_als   = b7  - yfit_als;
res_irls  = b7c - yfit_irls;

%% 95% prediction interval for ALS (unweighted)
m7    = length(b7);
p7    = size(A7,2);
s2_als = sum(res_als.^2)/(m7-p7);
t95   = tinv(0.975, m7-p7);
% Leverage-based CI width
AtA_inv = pinv(A7'*A7);
ci_als  = zeros(m7,1);
for ii=1:m7, ci_als(ii)=t95*sqrt(s2_als*(1+A7(ii,:)*AtA_inv*A7(ii,:)')); end

%% 95% prediction interval for IRLS (weighted)
W7    = diag(w7);
AW7   = A7'*W7;
s2_irls = sum(w7.*res_irls.^2)/(sum(w7)-p7);
AtWA_inv = pinv(AW7*A7);
ci_irls  = zeros(m7,1);
for ii=1:m7, ci_irls(ii)=t95*sqrt(s2_irls*(1+A7(ii,:)*AtWA_inv*A7(ii,:)')); end

%% Identify outlier index (lag-0: first entry of b, largest contamination)
[~,outlier_idx] = max(abs(res_als));   % should be index 1 (lag-0)

%% Projected x-axis: A*theta projected to scalar for plotting
% Use the predicted value yfit as x (adjusted predictor, fitlm convention)
x_als  = yfit_als;
x_irls = yfit_irls;

%% Sort for smooth CI plot
[xs_als,  sidx_als]  = sort(x_als);
[xs_irls, sidx_irls] = sort(x_irls);

%% ---- Figure 7: IRLSfit ----
figure(7); clf; set(gcf,'Position',[50 50 1000 420],'Name','Fig7: IRLSfit');

subplot(1,2,1); hold on; grid on; box on;
%% 95% CI band
fill([xs_als; flipud(xs_als)],...
     [b7(sidx_als)-ci_als(sidx_als); flipud(b7(sidx_als)+ci_als(sidx_als))],...
     [0.85 0.85 1],'EdgeColor','none','DisplayName','95% conf. bounds');
%% normal data points (blue cross)
nrm_idx = setdiff(1:m7, outlier_idx);
plot(x_als(nrm_idx), b7(nrm_idx), 'bx','MarkerSize',8,'LineWidth',1.5,...
     'DisplayName','Adjusted data');
%% outlier (red circle)
plot(x_als(outlier_idx), b7(outlier_idx), 'ro','MarkerSize',10,'LineWidth',2,...
     'MarkerFaceColor',[1 0.6 0.6],'DisplayName',sprintf('Outlier (lag-%d)',outlier_idx-1));
%% fit line
plot(xs_als, xs_als, 'b-','LineWidth',1.8,...
     'DisplayName',sprintf('Fit: y=%.4f x+%.4f',1,0));
xlabel('Adjusted $\mathbf{A}\hat{\theta}_{\rm ALS}$','Interpreter','latex','FontSize',12);
ylabel('Adjusted $\mathbf{b}$','Interpreter','latex','FontSize',12);
title(sprintf('ALS  ($\\hat{Q}=%.3f$, $\\hat{R}=%.3f$)',Q_als7,R_als7),'FontSize',12);
legend('Location','northwest','FontSize',9);

subplot(1,2,2); hold on; grid on; box on;
fill([xs_irls; flipud(xs_irls)],...
     [b7c(sidx_irls)-ci_irls(sidx_irls); flipud(b7c(sidx_irls)+ci_irls(sidx_irls))],...
     [1 0.88 0.88],'EdgeColor','none','DisplayName','95% conf. bounds');
plot(x_irls(nrm_idx), b7c(nrm_idx), 'bx','MarkerSize',8,'LineWidth',1.5,...
     'DisplayName','Adjusted data');
%% After cleaning, lag-0 outlier is removed; show its original value for reference
plot(x_irls(1), b7(1), 'rs','MarkerSize',10,'LineWidth',2,...
     'MarkerFaceColor',[1 0.6 0.6],'DisplayName',sprintf('Removed lag-0 (raw=%.1f)',b7(1)));
plot(xs_irls, xs_irls, 'r-','LineWidth',1.8,...
     'DisplayName',sprintf('Fit: $\\hat{Q}=%.3f$, $\\hat{R}=%.3f$',Q_irls7,R_irls7));
xlabel('Adjusted $\mathbf{A}\hat{\theta}_{\rm ALS-IRLS}$','Interpreter','latex','FontSize',12);
ylabel('Adjusted $\mathbf{b}_{\rm clean}$','Interpreter','latex','FontSize',12);
title(sprintf('ALS-IRLS  ($\\hat{Q}=%.3f$, $\\hat{R}=%.3f$)',Q_irls7,R_irls7),'FontSize',12);
legend('Location','northwest','FontSize',9);

sgtitle(sprintf(['Fitting comparison at MC trial 6, batch 1  '...
    '(outlier lag-0: raw=%.1f, clean=%.2f, true C_{e,0}=%.2f)'],...
    b7(1),b7c(1),Ce0c),'FontSize',11);

%% ---- Figure 8: IRLSfitweights ----
figure(8); clf; set(gcf,'Position',[50 500 560 360],'Name','Fig8: IRLSfitweights');
hold on; grid on; box on;

%% Identify outlier observations by weight < 0.5
is_outlier_w = w7 < 0.5;
bh8 = bar(1:m7, w7, 0.7);
bh8.FaceColor = 'flat';
for ii=1:m7
    if is_outlier_w(ii)
        bh8.CData(ii,:) = [0.5 0 0.5];   % purple (outlier)
    else
        bh8.CData(ii,:) = [0.2 0.4 0.8]; % blue (normal)
    end
end
hl8=yline(1,'k--','LineWidth',1.2); hl8.HandleVisibility='off';
hl8b=yline(0.5,'k:','LineWidth',1.2); hl8b.HandleVisibility='off';
text(m7*0.65, 1.04, 'w=1 (unit weight)','FontSize',9,'Color',[0.3 0.3 0.3]);
text(m7*0.65, 0.54, 'w=0.5 (threshold)','FontSize',9,'Color',[0.3 0.3 0.3]);
xticks(1:m7);
xticklabels(arrayfun(@(j)sprintf('lag-%d',j-1),1:m7,'UniformOutput',false));
xtickangle(45);
xlabel('Autocovariance entry','FontSize',12);
ylabel('IRLS Huber weight $w_j$','Interpreter','latex','FontSize',12);
% legend patches
h1=patch(NaN,NaN,[0.2 0.4 0.8]); h2=patch(NaN,NaN,[0.5 0 0.5]);
legend([h1,h2],{'Normal (w\geq0.5)','Downweighted (w<0.5)'},...
       'Location','south','FontSize',11);
title(sprintf('IRLS Huber weights — MC trial 6, batch 1  (outlier: lag-0, w=%.3f)',w7(1)),...
    'FontSize',11);
ylim([0 1.2]);

fprintf('\n=== MC trial 6, batch 1 estimates ===\n');
fprintf('  ALS:      Q=%.4f  R=%.4f\n',Q_als7,R_als7);
fprintf('  ALS-IRLS: Q=%.4f  R=%.4f\n',Q_irls7,R_irls7);
fprintf('  True:     Q=%.1f    R=%.1f\n',Q_true,R_true);
fprintf('  Outlier:  lag-0 weight = %.4f\n',w7(1));
fprintf('  b_LS[0] raw=%.2f  clean=%.2f  true Ce0=%.2f\n',b7(1),b7c(1),Ce0c);

fprintf('\nAll done. Figures 1-8 ready.\n');
