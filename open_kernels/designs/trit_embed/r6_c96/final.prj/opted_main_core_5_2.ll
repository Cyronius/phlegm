; ModuleID = '/home/cyrus/code/phlegm-trit-embed/open_kernels/designs/trit_embed/build_L3_cap128_d2_cycles96/final.prj/peano-linked_main_core_5_2.ll'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p:20:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-f32:32:32-i64:32-f64:32-a:0:32-n32"
target triple = "aie2p"

@buf_167 = global [16 x i32] zeroinitializer
@buf_166 = external global [1024 x bfloat]
@buf_165 = external global [32 x float]
@buf_164 = external global [3072 x i8]
@buf_163 = external global [4096 x i8]
@buf_162 = external global [768 x float]
@buf_161 = external global [3072 x bfloat]
@buf_160 = external global [3072 x float]
@stream_a1_4_cons_buff_1 = external global [4096 x i8]
@stream_a1_4_cons_buff_0 = external global [4096 x i8]
@stream_b1_4_cons_buff_1 = external global [4096 x i8]
@stream_b1_4_cons_buff_0 = external global [4096 x i8]
@y5_0_buff_0 = external global [9984 x i8]

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.aie2p.acquire(i32, i32) #0

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.aie2p.release(i32, i32) #0

declare void @f2_pre(ptr, ptr, ptr, ptr, ptr, i32) local_unnamed_addr

declare void @f2_step(ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32) local_unnamed_addr

declare void @f2_out(ptr, ptr, i32) local_unnamed_addr

define void @core_5_2() local_unnamed_addr {
  br label %.preheader11

.preheader11:                                     ; preds = %0, %102
  %1 = phi i32 [ 0, %0 ], [ %96, %102 ]
  %2 = phi i32 [ 0, %0 ], [ %93, %102 ]
  %3 = phi i64 [ 0, %0 ], [ %103, %102 ]
  br label %4

4:                                                ; preds = %16, %.preheader11
  %5 = phi i32 [ %1, %.preheader11 ], [ %37, %16 ]
  %6 = phi i32 [ %2, %.preheader11 ], [ %34, %16 ]
  %7 = phi i64 [ 0, %.preheader11 ], [ %38, %16 ]
  tail call void @llvm.aie2p.acquire(i32 53, i32 -1)
  %cond = icmp eq i32 %6, 1
  %spec.select = select i1 %cond, ptr @stream_a1_4_cons_buff_1, ptr @stream_a1_4_cons_buff_0
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  %cond1 = icmp eq i32 %5, 1
  %8 = select i1 %cond1, ptr @stream_b1_4_cons_buff_1, ptr @stream_b1_4_cons_buff_0
  tail call void @f2_pre(ptr nonnull %spec.select, ptr nonnull %8, ptr nonnull @buf_160, ptr nonnull @buf_166, ptr nonnull @buf_167, i32 20)
  tail call void @llvm.aie2p.release(i32 52, i32 1)
  %9 = add i32 %6, 1
  %10 = icmp sgt i32 %9, 1
  %11 = select i1 %10, i32 0, i32 %9
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  %12 = add i32 %5, 1
  %13 = icmp sgt i32 %12, 1
  %14 = select i1 %13, i32 0, i32 %12
  %15 = icmp samesign ult i64 %7, 24
  br i1 %15, label %16, label %.preheader

16:                                               ; preds = %4
  tail call void @llvm.aie2p.acquire(i32 53, i32 -1)
  %cond.1 = icmp eq i32 %11, 1
  %spec.select.1 = select i1 %cond.1, ptr @stream_a1_4_cons_buff_1, ptr @stream_a1_4_cons_buff_0
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  %cond1.1 = icmp eq i32 %14, 1
  %17 = select i1 %cond1.1, ptr @stream_b1_4_cons_buff_1, ptr @stream_b1_4_cons_buff_0
  tail call void @f2_pre(ptr nonnull %spec.select.1, ptr nonnull %17, ptr nonnull @buf_160, ptr nonnull @buf_166, ptr nonnull @buf_167, i32 20)
  tail call void @llvm.aie2p.release(i32 52, i32 1)
  %18 = add i32 %11, 1
  %19 = icmp sgt i32 %18, 1
  %20 = select i1 %19, i32 0, i32 %18
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  %21 = add i32 %14, 1
  %22 = icmp sgt i32 %21, 1
  %23 = select i1 %22, i32 0, i32 %21
  tail call void @llvm.aie2p.acquire(i32 53, i32 -1)
  %cond.2 = icmp eq i32 %20, 1
  %spec.select.2 = select i1 %cond.2, ptr @stream_a1_4_cons_buff_1, ptr @stream_a1_4_cons_buff_0
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  %cond1.2 = icmp eq i32 %23, 1
  %24 = select i1 %cond1.2, ptr @stream_b1_4_cons_buff_1, ptr @stream_b1_4_cons_buff_0
  tail call void @f2_pre(ptr nonnull %spec.select.2, ptr nonnull %24, ptr nonnull @buf_160, ptr nonnull @buf_166, ptr nonnull @buf_167, i32 20)
  tail call void @llvm.aie2p.release(i32 52, i32 1)
  %25 = add i32 %20, 1
  %26 = icmp sgt i32 %25, 1
  %27 = select i1 %26, i32 0, i32 %25
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  %28 = add i32 %23, 1
  %29 = icmp sgt i32 %28, 1
  %30 = select i1 %29, i32 0, i32 %28
  tail call void @llvm.aie2p.acquire(i32 53, i32 -1)
  %cond.3 = icmp eq i32 %27, 1
  %spec.select.3 = select i1 %cond.3, ptr @stream_a1_4_cons_buff_1, ptr @stream_a1_4_cons_buff_0
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  %cond1.3 = icmp eq i32 %30, 1
  %31 = select i1 %cond1.3, ptr @stream_b1_4_cons_buff_1, ptr @stream_b1_4_cons_buff_0
  tail call void @f2_pre(ptr nonnull %spec.select.3, ptr nonnull %31, ptr nonnull @buf_160, ptr nonnull @buf_166, ptr nonnull @buf_167, i32 20)
  tail call void @llvm.aie2p.release(i32 52, i32 1)
  %32 = add i32 %27, 1
  %33 = icmp sgt i32 %32, 1
  %34 = select i1 %33, i32 0, i32 %32
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  %35 = add i32 %30, 1
  %36 = icmp sgt i32 %35, 1
  %37 = select i1 %36, i32 0, i32 %35
  %38 = add nuw nsw i64 %7, 4
  br label %4

.preheader:                                       ; preds = %4, %99
  %39 = phi i32 [ %96, %99 ], [ %14, %4 ]
  %40 = phi i32 [ %93, %99 ], [ %11, %4 ]
  %41 = phi i64 [ %100, %99 ], [ 0, %4 ]
  tail call void @llvm.aie2p.acquire(i32 48, i32 -1)
  br label %42

42:                                               ; preds = %42, %.preheader
  %43 = phi i32 [ %39, %.preheader ], [ %60, %42 ]
  %44 = phi i32 [ %40, %.preheader ], [ %57, %42 ]
  %45 = phi i64 [ 0, %.preheader ], [ %61, %42 ]
  tail call void @llvm.aie2p.acquire(i32 53, i32 -1)
  %cond2 = icmp eq i32 %44, 1
  %spec.select9 = select i1 %cond2, ptr @stream_a1_4_cons_buff_1, ptr @stream_a1_4_cons_buff_0
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  %cond3 = icmp eq i32 %43, 1
  %46 = select i1 %cond3, ptr @stream_b1_4_cons_buff_1, ptr @stream_b1_4_cons_buff_0
  tail call void @f2_step(ptr nonnull %spec.select9, ptr nonnull %46, ptr nonnull @buf_160, ptr nonnull @buf_161, ptr nonnull @buf_162, ptr nonnull @buf_165, ptr nonnull @buf_166, ptr nonnull @buf_163, ptr nonnull @buf_164, ptr nonnull @y5_0_buff_0, ptr nonnull @buf_167, i32 1)
  tail call void @llvm.aie2p.release(i32 52, i32 1)
  %47 = add i32 %44, 1
  %48 = icmp sgt i32 %47, 1
  %49 = select i1 %48, i32 0, i32 %47
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  %50 = add i32 %43, 1
  %51 = icmp sgt i32 %50, 1
  %52 = select i1 %51, i32 0, i32 %50
  %53 = or disjoint i64 %45, 1
  tail call void @llvm.aie2p.acquire(i32 53, i32 -1)
  %cond2.1 = icmp eq i32 %49, 1
  %spec.select9.1 = select i1 %cond2.1, ptr @stream_a1_4_cons_buff_1, ptr @stream_a1_4_cons_buff_0
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  %cond3.1 = icmp eq i32 %52, 1
  %54 = select i1 %cond3.1, ptr @stream_b1_4_cons_buff_1, ptr @stream_b1_4_cons_buff_0
  tail call void @f2_step(ptr nonnull %spec.select9.1, ptr nonnull %54, ptr nonnull @buf_160, ptr nonnull @buf_161, ptr nonnull @buf_162, ptr nonnull @buf_165, ptr nonnull @buf_166, ptr nonnull @buf_163, ptr nonnull @buf_164, ptr nonnull @y5_0_buff_0, ptr nonnull @buf_167, i32 1)
  tail call void @llvm.aie2p.release(i32 52, i32 1)
  %55 = add i32 %49, 1
  %56 = icmp sgt i32 %55, 1
  %57 = select i1 %56, i32 0, i32 %55
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  %58 = add i32 %52, 1
  %59 = icmp sgt i32 %58, 1
  %60 = select i1 %59, i32 0, i32 %58
  %61 = add nuw nsw i64 %45, 2
  %62 = icmp samesign ult i64 %53, 13
  br i1 %62, label %42, label %63

63:                                               ; preds = %42
  tail call void @llvm.aie2p.release(i32 49, i32 1)
  br label %64

64:                                               ; preds = %64, %63
  %65 = phi i32 [ %60, %63 ], [ %96, %64 ]
  %66 = phi i32 [ %57, %63 ], [ %93, %64 ]
  %67 = phi i64 [ 0, %63 ], [ %97, %64 ]
  tail call void @llvm.aie2p.acquire(i32 53, i32 -1)
  %cond4 = icmp eq i32 %66, 1
  %spec.select10 = select i1 %cond4, ptr @stream_a1_4_cons_buff_1, ptr @stream_a1_4_cons_buff_0
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  %cond5 = icmp eq i32 %65, 1
  %68 = select i1 %cond5, ptr @stream_b1_4_cons_buff_1, ptr @stream_b1_4_cons_buff_0
  tail call void @f2_step(ptr nonnull %spec.select10, ptr nonnull %68, ptr nonnull @buf_160, ptr nonnull @buf_161, ptr nonnull @buf_162, ptr nonnull @buf_165, ptr nonnull @buf_166, ptr nonnull @buf_163, ptr nonnull @buf_164, ptr nonnull @y5_0_buff_0, ptr nonnull @buf_167, i32 1)
  tail call void @llvm.aie2p.release(i32 52, i32 1)
  %69 = add i32 %66, 1
  %70 = icmp sgt i32 %69, 1
  %71 = select i1 %70, i32 0, i32 %69
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  %72 = add i32 %65, 1
  %73 = icmp sgt i32 %72, 1
  %74 = select i1 %73, i32 0, i32 %72
  tail call void @llvm.aie2p.acquire(i32 53, i32 -1)
  %cond4.1 = icmp eq i32 %71, 1
  %spec.select10.1 = select i1 %cond4.1, ptr @stream_a1_4_cons_buff_1, ptr @stream_a1_4_cons_buff_0
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  %cond5.1 = icmp eq i32 %74, 1
  %75 = select i1 %cond5.1, ptr @stream_b1_4_cons_buff_1, ptr @stream_b1_4_cons_buff_0
  tail call void @f2_step(ptr nonnull %spec.select10.1, ptr nonnull %75, ptr nonnull @buf_160, ptr nonnull @buf_161, ptr nonnull @buf_162, ptr nonnull @buf_165, ptr nonnull @buf_166, ptr nonnull @buf_163, ptr nonnull @buf_164, ptr nonnull @y5_0_buff_0, ptr nonnull @buf_167, i32 1)
  tail call void @llvm.aie2p.release(i32 52, i32 1)
  %76 = add i32 %71, 1
  %77 = icmp sgt i32 %76, 1
  %78 = select i1 %77, i32 0, i32 %76
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  %79 = add i32 %74, 1
  %80 = icmp sgt i32 %79, 1
  %81 = select i1 %80, i32 0, i32 %79
  tail call void @llvm.aie2p.acquire(i32 53, i32 -1)
  %cond4.2 = icmp eq i32 %78, 1
  %spec.select10.2 = select i1 %cond4.2, ptr @stream_a1_4_cons_buff_1, ptr @stream_a1_4_cons_buff_0
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  %cond5.2 = icmp eq i32 %81, 1
  %82 = select i1 %cond5.2, ptr @stream_b1_4_cons_buff_1, ptr @stream_b1_4_cons_buff_0
  tail call void @f2_step(ptr nonnull %spec.select10.2, ptr nonnull %82, ptr nonnull @buf_160, ptr nonnull @buf_161, ptr nonnull @buf_162, ptr nonnull @buf_165, ptr nonnull @buf_166, ptr nonnull @buf_163, ptr nonnull @buf_164, ptr nonnull @y5_0_buff_0, ptr nonnull @buf_167, i32 1)
  tail call void @llvm.aie2p.release(i32 52, i32 1)
  %83 = add i32 %78, 1
  %84 = icmp sgt i32 %83, 1
  %85 = select i1 %84, i32 0, i32 %83
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  %86 = add i32 %81, 1
  %87 = icmp sgt i32 %86, 1
  %88 = select i1 %87, i32 0, i32 %86
  %89 = or disjoint i64 %67, 3
  tail call void @llvm.aie2p.acquire(i32 53, i32 -1)
  %cond4.3 = icmp eq i32 %85, 1
  %spec.select10.3 = select i1 %cond4.3, ptr @stream_a1_4_cons_buff_1, ptr @stream_a1_4_cons_buff_0
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  %cond5.3 = icmp eq i32 %88, 1
  %90 = select i1 %cond5.3, ptr @stream_b1_4_cons_buff_1, ptr @stream_b1_4_cons_buff_0
  tail call void @f2_step(ptr nonnull %spec.select10.3, ptr nonnull %90, ptr nonnull @buf_160, ptr nonnull @buf_161, ptr nonnull @buf_162, ptr nonnull @buf_165, ptr nonnull @buf_166, ptr nonnull @buf_163, ptr nonnull @buf_164, ptr nonnull @y5_0_buff_0, ptr nonnull @buf_167, i32 1)
  tail call void @llvm.aie2p.release(i32 52, i32 1)
  %91 = add i32 %85, 1
  %92 = icmp sgt i32 %91, 1
  %93 = select i1 %92, i32 0, i32 %91
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  %94 = add i32 %88, 1
  %95 = icmp sgt i32 %94, 1
  %96 = select i1 %95, i32 0, i32 %94
  %97 = add nuw nsw i64 %67, 4
  %98 = icmp samesign ult i64 %89, 95
  br i1 %98, label %64, label %99

99:                                               ; preds = %64
  %100 = add nuw nsw i64 %41, 1
  %101 = icmp samesign ult i64 %41, 2
  br i1 %101, label %.preheader, label %102

102:                                              ; preds = %99
  tail call void @llvm.aie2p.acquire(i32 48, i32 -1)
  tail call void @f2_out(ptr nonnull @buf_160, ptr nonnull @y5_0_buff_0, i32 0)
  tail call void @llvm.aie2p.release(i32 49, i32 1)
  tail call void @llvm.aie2p.acquire(i32 48, i32 -1)
  tail call void @f2_out(ptr nonnull @buf_160, ptr nonnull @y5_0_buff_0, i32 1)
  tail call void @llvm.aie2p.release(i32 49, i32 1)
  %103 = add nuw nsw i64 %3, 1
  %.not = icmp eq i64 %103, 9223372036854775807
  br i1 %.not, label %104, label %.preheader11

104:                                              ; preds = %102
  ret void
}

attributes #0 = { mustprogress nocallback nofree nosync nounwind willreturn }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
