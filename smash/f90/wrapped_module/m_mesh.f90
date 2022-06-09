!%    This module `m_mesh` encapsulates all SMASH mesh (type, subroutines, functions)
module m_mesh
    
    use m_common, only: dp, lchar
    use m_setup, only: SetupDT
    
    implicit none
    
    public :: MeshDT
    
    !%      MeshDT type:
    !%
    !%      ====================    ==========================================================
    !%      `args`                  Description
    !%      ====================    ==========================================================

    !%      ====================    ==========================================================
    
    type :: MeshDT
    
        integer :: nbx
        integer :: nby
        integer :: nbc
        integer :: xll
        integer :: yll
        integer, dimension(:,:), allocatable :: flow
        integer, dimension(:,:), allocatable :: drained_area
        
        character(20), dimension(:), allocatable :: code
        real(dp), dimension(:), allocatable :: area

    end type MeshDT
    
    contains
    
        subroutine MeshDT_initialise(mesh, setup, nbx, nby, nbc)
        
            implicit none
            
            type(SetupDT), intent(in) :: setup
            type(MeshDT), intent(inout) :: mesh
            integer, intent(in) :: nbx, nby, nbc
            
            mesh%nbx = nbx
            mesh%nby = nby
            mesh%nbc = nbc
            
            allocate(mesh%flow(mesh%nbx, mesh%nby)) 
            mesh%flow = -99
            allocate(mesh%drained_area(mesh%nbx, mesh%nby)) 
            mesh%drained_area = -99
            
            allocate(mesh%code(mesh%nbc))
            allocate(mesh%area(mesh%nbc))
            
        end subroutine MeshDT_initialise

end module m_mesh
