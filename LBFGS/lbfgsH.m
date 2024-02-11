
function r=lbfgsH(actset, g, S, Y, RHO, n, start_index, gamma, M)
% lbfgs two-loop function
if ~isempty(actset)
    %g=g(actset);
    S=S(actset,:);
    Y=Y(actset,:);
end
q=g;
if(n < M)
    for i = n : -1 : 1
        xi(i) = RHO(i) * dot(S(:,i),q);
        q = q- xi(i)*Y(:,i);
    end
    r = gamma*q;
    for i = 1 : n
        omega = RHO(i) * dot(r, Y(:,i));
        r =  r+ (xi(i) - omega)*S(:,i);
    end
    return;
end

for i = start_index + M - 1 : -1 : start_index
    ind = i;
    if(ind > M)
        ind = ind - M;
    end
    xi(ind) = RHO(ind) * dot(S(:,ind), q);
    q = q-xi(ind)*Y(:,ind);
end
r =  gamma*q;
for i = start_index : start_index + M - 1
    ind = i;
    if(ind > M)
        ind = ind - M;
    end
    omega = RHO(ind) * dot(Y(:,ind), r);
    r = r+ (xi(ind) - omega)*S(:,ind);
end

