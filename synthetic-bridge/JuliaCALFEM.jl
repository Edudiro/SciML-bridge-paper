using LinearAlgebra

# TODO: refactor these names to something descriptive
MyVector = Union{LinearAlgebra.Transpose,Adjoint,Vector}
MyVecOrMat = Union{LinearAlgebra.Transpose,Adjoint,VecOrMat}


"""
    spring1e(k) -> Ke

Computes the element stiffness matrix `Ke` for a
spring element with stiffness `k`.
"""
function spring1e(k::Number)
    return [
        k -k;
        -k k
    ]
end

"""
    spring1s(k, u) -> fe

Computes the force `fe` for a spring element with stiffness
`k` and displacements `u`.
"""
function spring1s(k::Number, u::Vector)
    if length(u) != 2
        throw(ArgumentError("displacements for computing the spring force must" *
                            "be a vector of length 2"))
    end
    return k * (u[2] - u[1])
end

"""
    bar2e(ex, ey, elem_prop) -> Ke

Computes the element stiffness matrix `Ke` for a 2D bar element.
"""
function bar2e(
    ex::MyVector,
    ey::MyVector,
    elem_prop::MyVector,
)
    # Local element stiffness
    E = elem_prop[1];  A = elem_prop[2]

    dx = ex[2] - ex[1]
    dy = ey[2] - ey[1]
    L = sqrt(dx^2 + dy^2)

    k = E * A / L
    Kel = [
        k  -k
        -k  k
    ]

    # Cosines
    c = dx / L; s = dy / L

    # Global element stiffness
    G = [c s 0 0
         0 0 c s]

    return G' * Kel * G

end


"""
    bar2s(ex, ey, elem_prop, el_disp) -> N

Computes the sectional force (normal force) `N` for a 2D bar element.
"""
function bar2s(
    ex::MyVector,
    ey::MyVector,
    elem_prop::MyVector,
    el_disp::MyVector,
)
    E = elem_prop[1];  A = elem_prop[2]

    dx = ex[2] - ex[1]
    dy = ey[2] - ey[1]
    L = sqrt(dx^2 + dy^2)

    k = E * A / L

    # Cosines
    c = dx / L; s = dy / L

    # Compute end displacements in local coordinate system
    G = [c s 0 0
         0 0 c s]
    print(size((vec(el_disp))))
    u = G * vec(el_disp)
    return k * (u[2] - u[1])
end


"""
Compute the stiffness matrix for a two dimensional beam element.

:param list ex: element x coordinates [x1, x2]
:param list ey: element y coordinates [y1, y2]
:param list elem_prop: element properties [E, A, I], E - Young's modulus, A - Cross section area, I - Moment of inertia
:param list eq: distributed loads, local directions [qx, qy]
:return mat Ke: element stiffness matrix [6 x 6]
:return mat fe: element stiffness matrix [6 x 1] (if eq!=None)
"""
function beam2e(
    ex::MyVector,
    ey::MyVector,
    elem_prop::MyVector,
    eq::Union{MyVector,Nothing}=nothing,
)
    b = [[ex[2] - ex[1]],[ey[2] - ey[1]]]
    L = sqrt(b' * b)
    n = reshape(b' / L, 2, )

    E = elem_prop[1]
    A = elem_prop[2]
    I = elem_prop[3]

    qx = 0.0
    qy = 0.0
    if eq != nothing
        qx = eq[1]
        qy = eq[2]
    end

    Kle = [
        E * A / L       0.0           0.0     -E * A / L     0.0         0.0
        0.0     12 * E * I / L^3.0  6 * E * I / L^2.0     0.0  -12 * E * I / L^3.0  6 * E * I / L^2.0
        0.0     6 * E * I / L^2.0   4 * E * I / L       0.0  -6 * E * I / L^2.0   2 * E * I / L
        -E * A / L      0.0           0.0      E * A / L     0.0         0.0
        0.0    -12 * E * I / L^3.0 -6 * E * I / L^2.0     0.0   12 * E * I / L^3.0 -6 * E * I / L^2.0
        0.0     6 * E * I / L^2.0   2 * E * I / L       0.0   -6 * E * I / L^2.0  4 * E * I / L
    ]'

    fle = L * [qx / 2, qy / 2, qy * L / 12, qx / 2, qy / 2, -qy * L / 12]

    G = [
        n[1]   n[2]    0.0     0.0     0.0    0.0
        -n[2]  n[1]    0.0     0.0     0.0    0.0
        0.0    0.0     1.0     0.0     0.0    0.0
        0.0    0.0     0.0     n[1]    n[2]   0.0
        0.0    0.0     0.0     -n[2]   n[1]   0.0
        0.0    0.0     0.0     0.0     0.0    1.0
    ]'

    Ke = G' * Kle * G
    fe = G' * fle

    if eq === nothing
        return Ke
    else
        return Ke, fe
    end
end



"""
Compute the stiffness matrix for a two dimensional beam element with 4 degrees of freedom.

:param list ex: element x coordinates [x1, x2]
:param list ey: element y coordinates [y1, y2]
:param list elem_prop: element properties [E, A, I], E - Young's modulus, A - Cross section area, I - Moment of inertia
:param list eq: distributed loads, local directions [qx, qy]
:return mat Ke: element stiffness matrix [4 x 4]
:return mat fe: element stiffness matrix [4 x 1] (if eq!=None)
"""
function beam2e4dof(
    ex::MyVector,
    ey::MyVector,
    elem_prop::MyVector,
    eq::Union{MyVector,Nothing}=nothing,
)
    b = [[ex[2] - ex[1]],[ey[2] - ey[1]]]
    L = sqrt(b' * b)
    n = reshape(b' / L, 2, )

    E = elem_prop[1]
    A = elem_prop[2]
    I = elem_prop[3]

    qx = 0.0
    qy = 0.0
    if eq != nothing
        qx = eq[1]
        qy = eq[2]
    end

    Kle = [
        12 * E * I / L^3.0  6 * E * I / L^2.0     -12 * E * I / L^3.0  6 * E * I / L^2.0
        6 * E * I / L^2.0   4 * E * I / L       -6 * E * I / L^2.0   2 * E * I / L
        -12 * E * I / L^3.0 -6 * E * I / L^2.0     12 * E * I / L^3.0 -6 * E * I / L^2.0
        6 * E * I / L^2.0   2 * E * I / L       -6 * E * I / L^2.0  4 * E * I / L
    ]'

    fle = L * [qy / 2, qy * L / 12, qy / 2, -qy * L / 12]

    G = [
        n[1]    0.0   0.0    0.0
        0.0     1.0   0.0    0.0
        0.0     0.0   n[1]   0.0
        0.0     0.0   0.0    1.0
    ]'

    Ke = G' * Kle * G
    fe = G' * fle

    if eq === nothing
        return Ke
    else
        return Ke, fe
    end
end




"""
Compute section forces in two dimensional beam element (beam2e).

Parameters:

ex = [x1 x2]
ey = [y1 y2]        element node coordinates
elem_prop = [E A I]        element properties,
E:  Young's modulus
A:  cross section area
I:  moment of inertia
ed = [u1 ... u6]    element displacements
eq = [qx qy]        distributed loads, local directions
nelem_prop                 number of evaluation points ( default=2 )

Returns:

es = [ N1 V1 M1     section forces, local directions, in
N2 V2 M2     n points along the beam, dim(es)= n x 3
.........]

edi = [ u1 v1       element displacements, local directions,
u2 v2       in n points along the beam, dim(es)= n x 2
.......]
eci = [ x1      local x-coordinates of the evaluation
x2      points, (x1=0 and xn=L)
...]

"""
function beam2s(
    ex::MyVector,
    ey::MyVector,
    elem_prop::MyVector,
    el_disp::MyVector,
    eq::Union{MyVector,Nothing}=nothing,
    nelem_prop::Union{Number,Nothing}=nothing,
)
    EA = elem_prop[1] * elem_prop[2]
    EI = elem_prop[1] * elem_prop[3]

    dx = ex[2] - ex[1]
    dy = ey[2] - ey[1]
    L = sqrt(dx^2 + dy^2)

    b = [dx dy]
    n = b / L  # Kolla reshape

    qx = 0.0
    qy = 0.0

    if eq != nothing
        qx = eq[1]
        qy = eq[2]
    end

    ne = 2

    if nelem_prop != nothing
        ne = nelem_prop
    end

    #=
    C = [
        0   0   0    1   0   0
        0   0   0    0   0   1
        0   0   0    0   1   0
        L   0   0    1   0   0
        0   L^3  L^2 0   L   1
        0 3 * L^2 2 * L  0   1   0
    ]

    # n=b/L

    G = [
        n[1]  n[2]  0    0     0    0
        -n[2] n[1]  0    0     0    0
        0     0     1    0     0    0
        0     0     0    n[1]  n[2] 0
        0     0     0    -n[2] n[1] 0
        0     0     0    0     0    1
    ]'
    =#

    C = [
        0   0  0   1
        0   0  1   0
        L^3  L^2  L   1
        3 * L^2  2 * L  1   0
    ]

    # n=b/L

    G = [
        n[1]  0    0    0
        0     1    0    0
        0     0    n[1] 0
        0     0    0    1
    ]'

    M = inv(C) * map(
        -,
        G * el_disp,
        #[0 0 0 -qx * L^2 / (2 * EA) qy * L^4 / (24 * EI) qy * L^3 / (6 * EI)]'
        [0  0  qy * L^4 / (24 * EI)  qy * L^3 / (6 * EI)]'
    )

    #A = [M[1] M[4]]';
    #B = [M[2] M[3] M[5] M[6]]';
    B = [M[1] M[2] M[3] M[4]]';

    x = collect(0:L / (ne - 1):L);   zero = zeros(length(x)); one = ones(length(x));
    #u = [x one] * A - (x.^2) * qx / (2 * EA);
    #du = [one zero] * A - x * qx / EA;
    v = [x.^3 x.^2 x one] * B + (x.^4) * qy / (24 * EI);
    d2v = [6 * x 2 * one zero zero] * B + (x.^2) * qy / (2 * EI);
    d3v = [6 * one zero zero zero] * B + x * qy / EI;

    #N = EA * du
    M = EI * d2v
    V = -EI * d3v
    #edi = hcat(u, v)
    eci = x
    #es = hcat(N, V, M)

    # Hack
    N = V
    edi = v
    es = hcat(N, V, M)

    return (es, edi, eci)
end



"""
Compute section forces in two dimensional beam element (beam2e) with 4 dofs.

Parameters:

ex = [x1 x2]
ey = [y1 y2]        element node coordinates
elem_prop = [E A I]        element properties,
E:  Young's modulus
A:  cross section area
I:  moment of inertia
ed = [u1 ... u6]    element displacements
eq = [qx qy]        distributed loads, local directions
nelem_prop                 number of evaluation points ( default=2 )

Returns:

es = [ N1 V1 M1     section forces, local directions, in
N2 V2 M2     n points along the beam, dim(es)= n x 3
.........]

edi = [ u1 v1       element displacements, local directions,
u2 v2       in n points along the beam, dim(es)= n x 2
.......]
eci = [ x1      local x-coordinates of the evaluation
x2      points, (x1=0 and xn=L)
...]

"""
function beam2s4dof(
    ex::MyVector,
    ey::MyVector,
    elem_prop::MyVector,
    el_disp::MyVector,
    eq::Union{MyVector,Nothing}=nothing,
    nelem_prop::Union{Number,Nothing}=nothing,
)
    EA = elem_prop[1] * elem_prop[2]
    EI = elem_prop[1] * elem_prop[3]

    dx = ex[2] - ex[1]
    dy = ey[2] - ey[1]
    L = sqrt(dx^2 + dy^2)

    b = [dx dy]
    n = b / L  # Kolla reshape

    qx = 0.0
    qy = 0.0

    if eq != nothing
        qx = eq[1]
        qy = eq[2]
    end

    ne = 2

    if nelem_prop != nothing
        ne = nelem_prop
    end


    C = [
        0   0   0   1
        0   0   1   0
        L^3  L^2   L   1
        3 * L^2   2 * L   1   0
    ]

    # n=b/L

    G = [
        n[1]    0.0   0.0    0.0
        0.0     1.0   0.0    0.0
        0.0     0.0   n[1]   0.0
        0.0     0.0   0.0    1.0
    ]'

    B = inv(C) * map(
        -,
        G * el_disp,
        [0    0    qy * L^4 / (24 * EI)    qy * L^3 / (6 * EI)]'
    )

    x = collect(0:L / (ne - 1):L);   zero = zeros(length(x)); one = ones(length(x));
    v = [x.^3 x.^2 x one] * B + (x.^4) * qy / (24 * EI);
    d2v = [6 * x 2 * one zero zero] * B + (x.^2) * qy / (2 * EI);
    d3v = [6 * one zero zero zero] * B + x * qy / EI;

    M = EI * d2v
    V = -EI * d3v
    edi = v
    eci = x
    es = hcat(V, M)

    return (es, edi, eci)
end



"""
Assembles the the element stiffness matrix `Ke`
to the global stiffness matrix `K`.
"""
function assem(
    edof::Vector,
    K::AbstractMatrix,
    Ke::Matrix,
    f::Union{VecOrMat,Nothing}=nothing,
    fe::Union{Vector,Nothing}=nothing,
) # FIX THIS
    (nr, nc) = size(Ke)
    if nr != nc
        throw(DimensionMismatch("Stiffness matrix is not square (#rows=$nr #cols=$nc)"))
    elseif length(edof) != nr
        len_edof = length(edof)
        throw(DimensionMismatch("Mismatch between sizes in edof and Ke (edof($len_edof) Ke($nr,$nc)"))
    end
    K[edof, edof] += Ke
    if f != nothing && f != nothing
        f[edof] += fe
        return K, f
    end
    return K
end

"""
    solveq(K, f, bc, [symmetric=false]) -> a, fb

Solves the equation system Ka = f taking into account the
Dirichlet boundary conditions in the matrix `bc`. Returns the solution vector `a`
and reaction forces `fb`
If `symmetric` is set to `true`, the matrix will be factorized with Cholesky factorization.
"""
function solveq(K::AbstractMatrix, f::Array, bc::Matrix)
    if !(size(K, 1) == size(K, 2))
        throw(DimensionMismatch("matrix not square"))
    end
    n = size(K, 2)
    nrf = length(f)
    if n != nrf
        throw(DimensionMismatch("Mismatch between number of rows in the stiffness matrix and load vector (#rowsK=$n #rowsf=$nrf)"))
    end

    d_pres = convert(Vector{Int}, bc[:,1])   # prescribed dofs
    a_pres = bc[:,2] # corresponding prescribed dof values

    # Construct array holding all the free dofs
    b_free = collect(1:n)
    d_free = b_free[(!).(in(d_pres).(b_free))]

    # convert to a sparse mx and factorize for performance
    spar_K = sparse(K[d_free, d_free])
    fact_spar_K = factorize(spar_K)

    # Solve equation system and create full solution vector a
    a_free = fact_spar_K \ (f[d_free] - K[d_free, d_pres] * a_pres)

    an = vcat(a_pres, a_free)
    dn = vcat(d_pres, d_free)

    c = sort(tuple.(an, dn), by=x -> x[2])
    d = [x[1] for x in c]
    a = d


    # Compute boundary force = reaction force
    f_b = K * a - f;

    return a, f_b
end

"""
    extract(edof, a)

Extracts the element displacements from the global solution vector `a`
given an `edof` matrix. This assumes all elements to have the same
number of dofs.
"""
function extract(edof::MyVecOrMat, a::VecOrMat)
    neldofs, nel = size(edof);

    eldisp = zeros(neldofs, nel)

    for el = 1:nel
        eldisp[:, el] = a[edof[:, el]]
    end
    return eldisp
end

function extract2(edof::MyVecOrMat, a::VecOrMat)
    return eldisp = a[edof]

end
