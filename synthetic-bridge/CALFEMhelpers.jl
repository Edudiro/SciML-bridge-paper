# ------------------------------------------------------
# HELPER FUNCTIONS FOR CALFEM.
# ------------------------------------------------------

# Load libraries
include("JuliaCALFEM.jl")  # Core CALFEM library

MyVector = Union{LinearAlgebra.Transpose,Adjoint,Vector}

# Generate edof matrix for linear structure
function build_lin_edof(numdof::Int,numstep::Int)
    Edof=zeros(Int16, (numdof*2,numstep))
    for i =1:numstep
        Edof[:,i] = range(1+numdof*(i-1), stop=2*numdof+numdof*(i-1), length=2*numdof)
    end
    return Edof
end

# Generate bc matrix for linear structure
function build_lin_bc(elem_geo::Vector, support_pos::Vector, numdof::Int)
    support_ind = 0
    n_support = length(support_pos)
    bc = zeros(Int64, n_support, 2)

    for i=1:n_support
        support_ind = numdof * (findfirst(isequal(support_pos[i]), elem_geo) - 1) + 1
        bc[i,1] = support_ind
    end

    return bc
end

# Main function for building a linear girder bridge.
function build_girder_bridge(elem_geo::Vector, support_pos::Vector, ep::MyVector, eq=[0.0 0.0]', numdof=2)
    numstep = length(elem_geo)-1

    # Build edof matrix and bc matrix
    Edof = build_lin_edof(numdof, numstep)

    bc = build_lin_bc(elem_geo, support_pos, numdof)

    # Allocate space
    n_elements = size(Edof)[2]
    n_dofs = length(vec(Edof))
    K = zeros(Float32, n_dofs, n_dofs)
    f = zeros(Float32, n_dofs, 1)
    Ke = zeros(Float32, 4, 4)
    ex = zeros(Float32, 2)
    ey = [0.0 0.0]'; #Assume linear bridge

    for i in 1:n_elements
        ex[:] = @view elem_geo[i:i+1]
        Ke[:,:] = beam2e4dof(ex, ey, ep)
        K[:,:] = assem(Edof[:,i], K, Ke)
    end

    return K, f, bc, Edof
end

# Main function for subdividing a linear system.
function rebuild_linear_system(L::AbstractFloat, step, elem_geo, ep, bc, edof, f, numdof=3, EP = nothing)

    # Subdivide and rebuild system
    (NewEdof,numstep,segments) = rebuild_edof(L,step,elem_geo,numdof)

    # Rebuild compatibility matrix
    newBc=rebuild_bc(bc,edof,segments,numdof, elem_geo)

    # Simplification for linear system
    ey=[0,0]

    # Allocate new stiffness matrix, ex and force vector
    K2 = zeros(numstep*numdof, numstep*numdof);
    Ke2 = zeros(numdof*2, numdof*2)
    ex = zeros(Float32, 2)

    # Assemble new system
    for i = 1:numstep-1
        # Custom properties
        if EP != nothing
            ep=EP[i,:]
        end
        ex[:] = @view segments[i:i+1]
        Ke2[:,:] = beam2e4dof(ex, ey, ep)
        K2[:,:] = assem(NewEdof[:,i], K2, Ke2)
    end

    # Rebuild force vector
    f2=rebuild_f(f,edof,segments,numdof, elem_geo)

    return K2,f2,newBc,NewEdof,segments
end


# Subdivides system and rebuilds compatibility matrix. Works only for linear systems.
function rebuild_edof(L,step, elem_geo, numdof)
    numstep=(Int16(L/step)+1)
    segments=collect(Float32, 0:step:L)

    original_x=Set(elem_geo)
    @test (original_x ⊆ segments)

    NewEdof=zeros(Int16, (numdof*2,numstep-1))

    for i =1:numstep-1
        NewEdof[:,i] = range(1+numdof*(i-1), stop=2*numdof+numdof*(i-1), length=2*numdof)
    end

    return NewEdof,numstep,segments
end




# Returns the dof from the first system in the rebuild system
function get_newdof(dof,old_edof,segments,numdof,elem_geo)
    elem = Tuple(findfirst(isequal(dof),old_edof))# Finds coordinates of bc in old_edof, column is element number, row%3 is element type
    isEnd = Int(elem[1]>numdof)
    dofType = elem[1]%numdof #1 is horisontal, 2 is vertical, 0 is rotation
    if dofType==0
        dofType=numdof
    end
    elem_loc=elem_geo[elem[2]+isEnd]
    new_elem_index=findfirst(isequal(elem_loc),segments)
    new_dof=numdof*(new_elem_index-1)+dofType
    return new_dof
end


# Rebuilds the f vector with new dimensions and the old loads in correct places.
function rebuild_f(f,old_edof,segments,numdof,elem_geo)
    newf=zeros(Float32,length(segments)*numdof,1)
        for i=1:length(f)
            F=f[i]
            if F !=0.0
                dof=i
                new_dof = get_newdof(dof,old_edof,segments,numdof,elem_geo)
                newf[new_dof]=F
            end
        end
    return newf
end


# Translates boundary conditions to new system.
function rebuild_bc(bc,old_edof,segments,numdof,elem_geo)
    newBc=zeros(Int64,length(@view bc[:,1]),2)

        for i=1:length(@view bc[:,1])
            dof=bc[i,1]
            new_dof = get_newdof(dof,old_edof,segments,numdof,elem_geo)

            newBc[i,1]=new_dof
        end
        return newBc
    end




# Functions for plotting rebuilt system
function plotmoments(edof,a,segments,n_eval=21,message ="RebuiltFEM")
    el_disp = extract(edof, a);     # Matrix contining the displacement vectors for each element
    ey=[0,0]
    ecind=0
    fig=plot()
    numstep = length(segments)

    for j = 1:numstep-1
        ex[:] = @view segments[j:j+1]
        # Calculate influence line and store data
        es, edi, eci = beam2s(ex, ey, ep, el_disp[:,j], ey, n_eval)
        eci.+=ecind
        plot!(fig,eci,es[:,3],label=message,seriescolor=:red,linewidth=1.5)
        ecind=last(eci)
    end

    display(fig)

end

function plotmoments!(fig,edof,a,segments,n_eval=21,message ="RebuiltFEM")
    el_disp = extract(edof, a);     # Matrix contining the displacement vectors for each element
    ey=[0,0]
    ecind=0
    numstep = length(segments)

    for j = 1:numstep-1
        ex[:] = @view segments[j:j+1]
        # Calculate influence line and store data
        es, edi, eci = beam2s(ex, ey, ep, el_disp[:,j], ey, n_eval)
        eci.+=ecind
        plot!(fig,eci,es[:,3],label=message,seriescolor=:red,linewidth=1.5)
        ecind=last(eci)
    end

    display(fig)

end
