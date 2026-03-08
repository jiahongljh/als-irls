function D = symtran(n)
% Duplication matrix: D_n * vech(X) = vec(X)
D = zeros(n^2, n*(n+1)/2); col = 1;
for j = 1:n
    for i = j:n
        E = zeros(n); E(i,j) = 1;
        if i ~= j, E(j,i) = 1; end
        D(:,col) = E(:); col = col + 1;
    end
end
end
