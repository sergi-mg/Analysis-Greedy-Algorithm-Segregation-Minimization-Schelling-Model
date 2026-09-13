program TFG 
    implicit none
    integer :: i,j, time, ios, k_s_1
    double precision :: S, H, Moore_neighbourhood
    character*64 :: file_save, file_read
    integer, dimension(:,:), allocatable :: M_ini
    double precision, dimension(:,:), allocatable :: K_1
    !parameters
    character*32 :: sL, salpha, srho, stau, sseed
    integer :: L, seed
    double precision :: alpha, tau, rho
    external :: Moore_neighbourhood

    !Parameters reading

    !Order to execute the program:
    !.\name.exe L alpha tau rho seed
    !real (2,3,4), integer (1,4)

    call getarg(1 , sL)
    call getarg(2 , salpha)
    call getarg(3 , stau)
    call getarg(4 , srho)
    call getarg(5, sseed)

    READ (sL,*) L
    READ (salpha,*) alpha
    READ (stau,*) tau
    READ (srho,*) rho
    READ (sseed, *) seed

    !files

    file_save="../data_results_fortran/results_simulation.dat"
    open(15,file=trim(file_save))

    file_read="../matrix/matrix.dat"
    open(11, file=trim(file_read))

    !read the initial matrix
    allocate(M_ini(1:L,1:L))
    do i=1,L
        read(11,*,iostat=ios) (M_ini(i,j),j=1,L)
        if (ios /= 0) then
            write(*,*) "Error reading M at i=", i," and j=", j
            stop
        end if
    enddo
    close(11)



    !simulation

    !random seed
    call init_genrand(seed)

    !Moore Neighbourhood - kernel
    k_s_1=3 !kernel's matrix size (3 x 3)
    allocate(K_1(1:k_s_1,1:k_s_1))
    call kernel_creation(Moore_neighbourhood,k_s_1,K_1)


    !time evolution of the system


    call model(M_ini,L,alpha,tau,rho,S,H,time,K_1,k_s_1)


    !save the results

    write(15,"(e16.8,3X,e16.8,3X,I0)") S, H, time
    close(15)
    
end program

subroutine kernel_creation(funcio, kernel_size, kernel)
    !square kernel
    implicit none 
    integer :: kernel_size, center,i,j
    double precision :: funcio, kernel(1:kernel_size,1:kernel_size)
    center=int(kernel_size/2)+1
    do i=1,kernel_size
        do j=1,kernel_size
            if (i.ne.center.or.j.ne.center) then
                kernel(i,j)=funcio(center,center,i,j)
            endif
        enddo
    enddo
    kernel(center,center)=0.d0
    return
end subroutine

double precision function Moore_neighbourhood(x1,y1,x2,y2)
    implicit none
    integer :: x1,y1,x2,y2
    
    if (abs(x1-x2).lt.1.5d0.and.abs(y1-y2).lt.1.5d0) then
        Moore_neighbourhood=1.d0
    else
        Moore_neighbourhood=0.d0 
    endif

    return
end function


subroutine G_function(M,tau,alpha,long,rho_0,G,S,H,kernel,unhappy,vacants,&
    uh_count,kernel_size)
    !returns the values of S, H and G, two matrices with the positions of
    !the vacants and the unhappy agents, and the number of unhappy agents
    implicit none
    !input
    integer :: long, M(1:long,1:long), kernel_size, agents
    double precision :: tau, alpha, rho_0, kernel(1:kernel_size,1:kernel_size)
    !intern variables
    integer :: i,j,k,l,vacant_counter,unhappy_counter,limit
    double precision :: A, B, C, D
    !output
    integer :: unhappy(1:long**2,1:2), vacants(1:int(rho_0*long**2+0.1),1:2),&
    uh_count
    double precision :: G, S, H

    !unhappy matrix initialized
    do i=1,long**2
        unhappy(i,1)=0
        unhappy(i,2)=0
    enddo

    H=0.d0
    A=0.d0
    B=0.d0
    vacant_counter=0
    unhappy_counter=0
    limit=int(kernel_size/2)
    do i=1,long 
        do j=1,long
            if (M(i,j).eq.0) then
                vacant_counter=vacant_counter+1
                vacants(vacant_counter,1)=i
                vacants(vacant_counter,2)=j
            else
                C=0.d0
                D=0.d0
                do k=i-limit,i+limit
                    do l=j-limit,j+limit
                        if ((i.eq.k).and.(j.eq.l)) then
                            !none
                        else if (((k.lt.1).or.(k.gt.long)).or.((l.lt.1)&
                            .or.(l.gt.long))) then
                            !none
                        else
                            !S
                            A=A+((M(i,j)*M(k,l))**2.d0+M(i,j)*M(k,l))&
                            *kernel(k-(i-limit)+1,l-(j-limit)+1)
                            B=B+2*kernel(k-(i-limit)+1,l-(j-limit)+1)&
                            *(M(i,j)*M(k,l))**2.d0
                            !H
                            C=C+((M(i,j)*M(k,l))**2.d0-M(i,j)*M(k,l))&
                            *kernel(k-(i-limit)+1,l-(j-limit)+1)
                            D=D+2*kernel(k-(i-limit)+1,l-(j-limit)+1)&
                            *(M(i,j)*M(k,l))**2.d0
                        endif
                    enddo
                enddo
                if (D.ne.0) then
                    if (C/D.le.tau) then
                        H=H+1.d0
                    else
                        unhappy_counter=unhappy_counter+1
                        unhappy(unhappy_counter,1)=i
                        unhappy(unhappy_counter,2)=j
                    endif
                else
                    H=H+1.d0
                endif
            endif
        enddo
    enddo
    if (B.ne.0) then
        S=A/B
    else
        S=0.d0 
    endif
    H=H/((1.d0-rho_0)*long**2.d0)
    uh_count=unhappy_counter
    !G
    G=alpha*S-(1.d0-alpha)*H
    return
end subroutine

subroutine model(mat_ini, long, alpha, tau, rho_0, S_f, H_f, T_f,&
 kernel_matrix, kernel_size)
!given an initial matrix of vacant density rho_0, executes the
!algorithm for a certain alpha value and kernel and returns the
!final values of S and H and the total number of movements T_f
    implicit none
    !input variables
    integer :: long, mat_ini(1:long,1:long),kernel_size
    double precision :: alpha, tau, rho_0, &
    kernel_matrix(1:kernel_size,1:kernel_size)
    !output variables
    integer :: T_f
    double precision :: S_f, H_f
    !subroutine intern variables
    !matrices
    integer :: uh_list(1:long**2,1:2),&
    vacants(1:int(rho_0*long**2+0.1),1:2),&
    new_matrix(1:long,1:long), matrix(1:long,1:long), &
    uh_list_out(1:long**2,1:2),vacants_out(1:int(rho_0*long**2+0.1),1:2)
    double precision :: G_list(1:int(rho_0*long**2+0.1))
    !numbers
    integer :: N_uh_out, N_uh,i,j,k,l,global_iteration,N_vacant&
    ,vacant_min,rand_uh, N_uh_try
    double precision :: G_out, S_out, H_out, G_min, genrand_real3

    !definitions
    do i=1,long
        do j=1,long
            matrix(i,j)=mat_ini(i,j)
        enddo
    enddo

    N_vacant=int(rho_0*long**2+0.1)
    N_uh=1 !different from zero to enter the loop
    T_f=0 !number of movements
    !algorithm
    do global_iteration=1,1000
        if (global_iteration.eq.1000) then
            write(*,*)"Final state not reached for alpha=",alpha,&
            " and rho_0=",rho_0
        endif
        if (N_uh.eq.0) then
            exit
        endif        
        !how many unhappy agents? where? 
        call G_function(matrix,tau,alpha,long,rho_0,G_out,S_out,H_out,&
            kernel_matrix,uh_list,vacants,N_uh_out,kernel_size)
        if (N_uh_out.eq.0) then
            exit
        endif
        !movement selection
        N_uh=N_uh_out
        do i=1, N_uh_out
            !we chose an unhappy agent at random
            rand_uh=ceiling(genrand_real3()*N_uh)
            !we look for the most suitable movement
            do j=1, N_vacant
                !definition of the new matrix 
                do k=1,long
                    do l=1,long
                        new_matrix(k,l)=matrix(k,l)
                    enddo
                enddo
                new_matrix(vacants(j,1),vacants(j,2))=&
                matrix(uh_list(rand_uh,1),uh_list(rand_uh,2))
                new_matrix(uh_list(rand_uh,1),uh_list(rand_uh,2))=&
                matrix(vacants(j,1),vacants(j,2))
                !G calculus
                call G_function(new_matrix,tau,alpha,long,rho_0,G_out,&
                    S_out,H_out,kernel_matrix,uh_list_out,vacants_out,&
                    N_uh_try,kernel_size)
                !if the agent is unhappy we discard this movement (G>>1)
                do k=1,N_uh_try
                    if (uh_list_out(k,1).eq.vacants(j,1).and.&
                        uh_list_out(k,2).eq.vacants(j,2)) then
                        G_out=10.d0
                        exit
                    endif
                enddo
                !we save the value
                G_list(j)=G_out
            enddo
            !now we look for the minimum value of G
            G_min=10.d0
            do j=1, N_vacant
                if (G_list(j).lt.G_min) then
                    G_min=G_list(j)
                    vacant_min=j
                endif
            enddo
            if (G_min.ge.9.d0) then
                !the agent cannot move, we go to the following agent
                do k=1,2
                    uh_list(rand_uh,k)=uh_list(N_uh,k)
                    uh_list(N_uh,k)=0
                enddo
                N_uh=N_uh-1
                if (N_uh.eq.0) then
                    exit
                endif
            else
            !we execute the movement and restart the movement selection
                matrix(vacants(vacant_min,1),vacants(vacant_min,2))=&
                matrix(uh_list(rand_uh,1),uh_list(rand_uh,2))
                matrix(uh_list(rand_uh,1),uh_list(rand_uh,2))=0
                T_f=T_f+1
                exit
            endif
        enddo
    enddo
    call G_function(matrix,tau,alpha,long,rho_0,G_out,S_out,H_out,&
        kernel_matrix,uh_list,vacants,N_uh_out,kernel_size)
    S_f=S_out
    H_f=H_out
    return
end subroutine

!-----------------------------------------------------------------------
!mt19937ar (obtained in the Collective Phenomena and Phase Transitions
!course at Physics UB)

subroutine init_genrand(s)
      integer s
      integer N
      integer DONE
      integer ALLBIT_MASK
      parameter (N=624)
      parameter (DONE=123456789)
      integer mti,initialized
      integer mt(0:N-1)
      common /mt_state1/ mti,initialized
      common /mt_state2/ mt
      common /mt_mask1/ ALLBIT_MASK
!
      call mt_initln
      mt(0)=iand(s,ALLBIT_MASK)
      do 100 mti=1,N-1
        mt(mti)=1812433253*&
               ieor(mt(mti-1),ishft(mt(mti-1),-30))+mti
        mt(mti)=iand(mt(mti),ALLBIT_MASK)
  100 continue
      initialized=DONE
!
      return
      end
!-----------------------------------------------------------------------
!     initialize by an array with array-length
!     init_key is the array for initializing keys
!     key_length is its length
!-----------------------------------------------------------------------
      subroutine init_by_array(init_key,key_length)
      integer init_key(0:*)
      integer key_length
      integer N
      integer ALLBIT_MASK
      integer TOPBIT_MASK
      parameter (N=624)
      integer i,j,k
      integer mt(0:N-1)
      common /mt_state2/ mt
      common /mt_mask1/ ALLBIT_MASK
      common /mt_mask2/ TOPBIT_MASK
!
      call init_genrand(19650218)
      i=1
      j=0
      do 100 k=max(N,key_length),1,-1
        mt(i)=ieor(mt(i),ieor(mt(i-1),ishft(mt(i-1),-30))*1664525)&
                +init_key(j)+j
        mt(i)=iand(mt(i),ALLBIT_MASK)
        i=i+1
        j=j+1
        if(i.ge.N)then
          mt(0)=mt(N-1)
          i=1
        endif
        if(j.ge.key_length)then
          j=0
        endif
  100 continue
      do 200 k=N-1,1,-1
        mt(i)=ieor(mt(i),ieor(mt(i-1),ishft(mt(i-1),-30))*1566083941)-i
        mt(i)=iand(mt(i),ALLBIT_MASK)
        i=i+1
        if(i.ge.N)then
          mt(0)=mt(N-1)
          i=1
        endif
  200 continue
      mt(0)=TOPBIT_MASK
!
      return
      end
!-----------------------------------------------------------------------
!     generates a random number on [0,0xffffffff]-interval
!-----------------------------------------------------------------------
      function genrand_int32()
      integer genrand_int32
      integer N,M
      integer DONE
      integer UPPER_MASK,LOWER_MASK,MATRIX_A
      integer T1_MASK,T2_MASK
      parameter (N=624)
      parameter (M=397)
      parameter (DONE=123456789)
      integer mti,initialized
      integer mt(0:N-1)
      integer y,kk
      integer mag01(0:1)
      common /mt_state1/ mti,initialized
      common /mt_state2/ mt
      common /mt_mask3/ UPPER_MASK,LOWER_MASK,MATRIX_A,T1_MASK,T2_MASK
      common /mt_mag01/ mag01
!
      if(initialized.ne.DONE)then
        call init_genrand(21641)
      endif
!
      if(mti.ge.N)then
        do 100 kk=0,N-M-1
          y=ior(iand(mt(kk),UPPER_MASK),iand(mt(kk+1),LOWER_MASK))
          mt(kk)=ieor(ieor(mt(kk+M),ishft(y,-1)),mag01(iand(y,1)))
  100   continue
        do 200 kk=N-M,N-1-1
          y=ior(iand(mt(kk),UPPER_MASK),iand(mt(kk+1),LOWER_MASK))
          mt(kk)=ieor(ieor(mt(kk+(M-N)),ishft(y,-1)),mag01(iand(y,1)))
  200   continue
        y=ior(iand(mt(N-1),UPPER_MASK),iand(mt(0),LOWER_MASK))
        mt(kk)=ieor(ieor(mt(M-1),ishft(y,-1)),mag01(iand(y,1)))
        mti=0
      endif
!
      y=mt(mti)
      mti=mti+1
!
      y=ieor(y,ishft(y,-11))
      y=ieor(y,iand(ishft(y,7),T1_MASK))
      y=ieor(y,iand(ishft(y,15),T2_MASK))
      y=ieor(y,ishft(y,-18))
!
      genrand_int32=y
      return
      end
!-----------------------------------------------------------------------
!     generates a random number on [0,0x7fffffff]-interval
!-----------------------------------------------------------------------
      function genrand_int31()
      integer genrand_int31
      integer genrand_int32
      genrand_int31=int(ishft(genrand_int32(),-1))
      return
      end
!-----------------------------------------------------------------------
!     generates a random number on [0,1]-real-interval
!-----------------------------------------------------------------------
      function genrand_real1()
      double precision genrand_real1,r
      integer genrand_int32
      r=dble(genrand_int32())
      if(r.lt.0.d0)r=r+2.d0**32
      genrand_real1=r/4294967295.d0
      return
      end
!-----------------------------------------------------------------------
!     generates a random number on [0,1)-real-interval
!-----------------------------------------------------------------------
      function genrand_real2()
      double precision genrand_real2,r
      integer genrand_int32
      r=dble(genrand_int32())
      if(r.lt.0.d0)r=r+2.d0**32
      genrand_real2=r/4294967296.d0
      return
      end
!-----------------------------------------------------------------------
!     generates a random number on (0,1)-real-interval
!-----------------------------------------------------------------------
      function genrand_real3()
      double precision genrand_real3,r
      integer genrand_int32
      r=dble(genrand_int32())
      if(r.lt.0.d0)r=r+2.d0**32
      genrand_real3=(r+0.5d0)/4294967296.d0
      return
      end
!-----------------------------------------------------------------------
!     generates a random number on [0,1) with 53-bit resolution
!-----------------------------------------------------------------------
      function genrand_res53()
      double precision genrand_res53
      integer genrand_int32
      double precision a,b
      a=dble(ishft(genrand_int32(),-5))
      b=dble(ishft(genrand_int32(),-6))
      if(a.lt.0.d0)a=a+2.d0**32
      if(b.lt.0.d0)b=b+2.d0**32
      genrand_res53=(a*67108864.d0+b)/9007199254740992.d0
      return
      end
!-----------------------------------------------------------------------
!     initialize large number (over 32-bit constant number)
!-----------------------------------------------------------------------
      subroutine mt_initln
      integer ALLBIT_MASK
      integer TOPBIT_MASK
      integer UPPER_MASK,LOWER_MASK,MATRIX_A,T1_MASK,T2_MASK
      integer mag01(0:1)
      common /mt_mask1/ ALLBIT_MASK
      common /mt_mask2/ TOPBIT_MASK
      common /mt_mask3/ UPPER_MASK,LOWER_MASK,MATRIX_A,T1_MASK,T2_MASK
      common /mt_mag01/ mag01
!    TOPBIT_MASK = Z'80000000'
!    ALLBIT_MASK = Z'ffffffff'
!    UPPER_MASK  = Z'80000000'
!    LOWER_MASK  = Z'7fffffff'
!    MATRIX_A    = Z'9908b0df'
!    T1_MASK     = Z'9d2c5680'
!    T2_MASK     = Z'efc60000'
      TOPBIT_MASK=1073741824
      TOPBIT_MASK=ishft(TOPBIT_MASK,1)
      ALLBIT_MASK=2147483647
      ALLBIT_MASK=ior(ALLBIT_MASK,TOPBIT_MASK)
      UPPER_MASK=TOPBIT_MASK
      LOWER_MASK=2147483647
      MATRIX_A=419999967
      MATRIX_A=ior(MATRIX_A,TOPBIT_MASK)
      T1_MASK=489444992
      T1_MASK=ior(T1_MASK,TOPBIT_MASK)
      T2_MASK=1875247104
      T2_MASK=ior(T2_MASK,TOPBIT_MASK)
      mag01(0)=0
      mag01(1)=MATRIX_A
      return
      end