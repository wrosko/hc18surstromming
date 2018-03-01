function waitAD = WaitAtDestination(s_i, d_ij, t)
length_d_ij = length(d_ij);
s_i_matrix = repmat(s_i,1,length_d_ij);
waitAD = s_i_matrix - d_ij - t;
end