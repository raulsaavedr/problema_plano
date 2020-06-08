function solution = next(X, row, nr, whichrowstoplace)
    disp('Adentro de next');
    firstrow=X(row,:);           %Take the row
    disp('firstrow:'); disp(firstrow)
    onelocation=find(firstrow);  %And find the location of the ones
    disp('onelocation:'); disp(firstrow)
    leaves=onelocation(ismember(onelocation,whichrowstoplace));  %We are only interested in the ones that we still need
    disp('leaves:'); disp(leaves)
    if nr>length(leaves)         %You exhausted all possible tests, the row cannot be placed and therefore root must be bad
        solution=NaN;
    else                         %Otherwise leaves(nr) is the next (candidate) solution
        solution=[leaves(nr) first(X, row, whichrowstoplace(whichrowstoplace~=leaves(nr)))];
    end;
end
%{
comentarios generales de las evaluaciones en next
leaves almacena las posiciones en donde se encuentran los unos 
en la fila actual  0   1   1   1   0
2 3 4
el whichstoplace = [1 2 3 4 5]
pero para el siguiente llamado de first se extrae el elemento leaves(nr) de
el whichstoplace entonces whichstoplace = [1 3 4 5]
que para este caso es 2
entonce
solution = [2 first(X,)]
%}