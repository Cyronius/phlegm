; ModuleID = '/home/cyrus/code/phlegm-trit-embed/open_kernels/designs/trit_embed/build_L3_cap128_d2_cycles111/final.prj/peano-linked_main_core_1_4.ll'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p:20:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-f32:32:32-i64:32-f64:32-a:0:32-n32"
target triple = "aie2p"

@buf_55 = global [16 x i32] zeroinitializer
@buf_54 = external global [1024 x bfloat]
@buf_53 = external global [32 x float]
@buf_52 = external global [3072 x i8]
@buf_51 = external global [4096 x i8]
@buf_50 = external global [768 x float]
@buf_49 = external global [3072 x bfloat]
@buf_48 = external global [3072 x float]
@stream_a_6_cons_buff_1 = external global [4096 x i8]
@stream_a_6_cons_buff_0 = external global [4096 x i8]
@stream_b_6_cons_buff_1 = external global [4096 x i8]
@stream_b_6_cons_buff_0 = external global [4096 x i8]
@y1_2_buff_0 = external global [9984 x i8]

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.aie2p.acquire(i32, i32) #0

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.aie2p.release(i32, i32) #0

declare void @f2_pre(ptr, ptr, ptr, ptr, ptr, i32) local_unnamed_addr

declare void @f2_step(ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32) local_unnamed_addr

declare void @f2_out(ptr, ptr, i32) local_unnamed_addr

define void @core_1_4() local_unnamed_addr {
  br label %.preheader11

.preheader11:                                     ; preds = %0, %108
  %1 = phi i32 [ 0, %0 ], [ %102, %108 ]
  %2 = phi i32 [ 0, %0 ], [ %99, %108 ]
  %3 = phi i64 [ 0, %0 ], [ %109, %108 ]
  br label %4

4:                                                ; preds = %16, %.preheader11
  %5 = phi i32 [ %1, %.preheader11 ], [ %37, %16 ]
  %6 = phi i32 [ %2, %.preheader11 ], [ %34, %16 ]
  %7 = phi i64 [ 0, %.preheader11 ], [ %38, %16 ]
  tail call void @llvm.aie2p.acquire(i32 53, i32 -1)
  %cond = icmp eq i32 %6, 1
  %spec.select = select i1 %cond, ptr @stream_a_6_cons_buff_1, ptr @stream_a_6_cons_buff_0
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  %cond1 = icmp eq i32 %5, 1
  %8 = select i1 %cond1, ptr @stream_b_6_cons_buff_1, ptr @stream_b_6_cons_buff_0
  tail call void @f2_pre(ptr nonnull %spec.select, ptr nonnull %8, ptr nonnull @buf_48, ptr nonnull @buf_54, ptr nonnull @buf_55, i32 6)
  tail call void @llvm.aie2p.release(i32 52, i32 1)
  %9 = add i32 %6, 1
  %10 = icmp sgt i32 %9, 1
  %11 = select i1 %10, i32 0, i32 %9
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  %12 = add i32 %5, 1
  %13 = icmp sgt i32 %12, 1
  %14 = select i1 %13, i32 0, i32 %12
  %15 = icmp samesign ult i64 %7, 48
  br i1 %15, label %16, label %.preheader

16:                                               ; preds = %4
  tail call void @llvm.aie2p.acquire(i32 53, i32 -1)
  %cond.1 = icmp eq i32 %11, 1
  %spec.select.1 = select i1 %cond.1, ptr @stream_a_6_cons_buff_1, ptr @stream_a_6_cons_buff_0
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  %cond1.1 = icmp eq i32 %14, 1
  %17 = select i1 %cond1.1, ptr @stream_b_6_cons_buff_1, ptr @stream_b_6_cons_buff_0
  tail call void @f2_pre(ptr nonnull %spec.select.1, ptr nonnull %17, ptr nonnull @buf_48, ptr nonnull @buf_54, ptr nonnull @buf_55, i32 6)
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
  %spec.select.2 = select i1 %cond.2, ptr @stream_a_6_cons_buff_1, ptr @stream_a_6_cons_buff_0
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  %cond1.2 = icmp eq i32 %23, 1
  %24 = select i1 %cond1.2, ptr @stream_b_6_cons_buff_1, ptr @stream_b_6_cons_buff_0
  tail call void @f2_pre(ptr nonnull %spec.select.2, ptr nonnull %24, ptr nonnull @buf_48, ptr nonnull @buf_54, ptr nonnull @buf_55, i32 6)
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
  %spec.select.3 = select i1 %cond.3, ptr @stream_a_6_cons_buff_1, ptr @stream_a_6_cons_buff_0
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  %cond1.3 = icmp eq i32 %30, 1
  %31 = select i1 %cond1.3, ptr @stream_b_6_cons_buff_1, ptr @stream_b_6_cons_buff_0
  tail call void @f2_pre(ptr nonnull %spec.select.3, ptr nonnull %31, ptr nonnull @buf_48, ptr nonnull @buf_54, ptr nonnull @buf_55, i32 6)
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

.preheader:                                       ; preds = %4, %105
  %39 = phi i32 [ %102, %105 ], [ %14, %4 ]
  %40 = phi i32 [ %99, %105 ], [ %11, %4 ]
  %41 = phi i64 [ %106, %105 ], [ 0, %4 ]
  tail call void @llvm.aie2p.acquire(i32 48, i32 -1)
  br label %42

42:                                               ; preds = %42, %.preheader
  %43 = phi i32 [ %39, %.preheader ], [ %66, %42 ]
  %44 = phi i32 [ %40, %.preheader ], [ %63, %42 ]
  %45 = phi i64 [ 0, %.preheader ], [ %67, %42 ]
  tail call void @llvm.aie2p.acquire(i32 53, i32 -1)
  %cond2 = icmp eq i32 %44, 1
  %spec.select9 = select i1 %cond2, ptr @stream_a_6_cons_buff_1, ptr @stream_a_6_cons_buff_0
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  %cond3 = icmp eq i32 %43, 1
  %46 = select i1 %cond3, ptr @stream_b_6_cons_buff_1, ptr @stream_b_6_cons_buff_0
  tail call void @f2_step(ptr nonnull %spec.select9, ptr nonnull %46, ptr nonnull @buf_48, ptr nonnull @buf_49, ptr nonnull @buf_50, ptr nonnull @buf_53, ptr nonnull @buf_54, ptr nonnull @buf_51, ptr nonnull @buf_52, ptr nonnull @y1_2_buff_0, ptr nonnull @buf_55, i32 0)
  tail call void @llvm.aie2p.release(i32 52, i32 1)
  %47 = add i32 %44, 1
  %48 = icmp sgt i32 %47, 1
  %49 = select i1 %48, i32 0, i32 %47
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  %50 = add i32 %43, 1
  %51 = icmp sgt i32 %50, 1
  %52 = select i1 %51, i32 0, i32 %50
  tail call void @llvm.aie2p.acquire(i32 53, i32 -1)
  %cond2.1 = icmp eq i32 %49, 1
  %spec.select9.1 = select i1 %cond2.1, ptr @stream_a_6_cons_buff_1, ptr @stream_a_6_cons_buff_0
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  %cond3.1 = icmp eq i32 %52, 1
  %53 = select i1 %cond3.1, ptr @stream_b_6_cons_buff_1, ptr @stream_b_6_cons_buff_0
  tail call void @f2_step(ptr nonnull %spec.select9.1, ptr nonnull %53, ptr nonnull @buf_48, ptr nonnull @buf_49, ptr nonnull @buf_50, ptr nonnull @buf_53, ptr nonnull @buf_54, ptr nonnull @buf_51, ptr nonnull @buf_52, ptr nonnull @y1_2_buff_0, ptr nonnull @buf_55, i32 0)
  tail call void @llvm.aie2p.release(i32 52, i32 1)
  %54 = add i32 %49, 1
  %55 = icmp sgt i32 %54, 1
  %56 = select i1 %55, i32 0, i32 %54
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  %57 = add i32 %52, 1
  %58 = icmp sgt i32 %57, 1
  %59 = select i1 %58, i32 0, i32 %57
  tail call void @llvm.aie2p.acquire(i32 53, i32 -1)
  %cond2.2 = icmp eq i32 %56, 1
  %spec.select9.2 = select i1 %cond2.2, ptr @stream_a_6_cons_buff_1, ptr @stream_a_6_cons_buff_0
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  %cond3.2 = icmp eq i32 %59, 1
  %60 = select i1 %cond3.2, ptr @stream_b_6_cons_buff_1, ptr @stream_b_6_cons_buff_0
  tail call void @f2_step(ptr nonnull %spec.select9.2, ptr nonnull %60, ptr nonnull @buf_48, ptr nonnull @buf_49, ptr nonnull @buf_50, ptr nonnull @buf_53, ptr nonnull @buf_54, ptr nonnull @buf_51, ptr nonnull @buf_52, ptr nonnull @y1_2_buff_0, ptr nonnull @buf_55, i32 0)
  tail call void @llvm.aie2p.release(i32 52, i32 1)
  %61 = add i32 %56, 1
  %62 = icmp sgt i32 %61, 1
  %63 = select i1 %62, i32 0, i32 %61
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  %64 = add i32 %59, 1
  %65 = icmp sgt i32 %64, 1
  %66 = select i1 %65, i32 0, i32 %64
  %67 = add nuw nsw i64 %45, 3
  %68 = icmp samesign ult i64 %45, 15
  br i1 %68, label %42, label %69

69:                                               ; preds = %42
  tail call void @llvm.aie2p.release(i32 49, i32 1)
  br label %70

70:                                               ; preds = %70, %69
  %71 = phi i32 [ %66, %69 ], [ %102, %70 ]
  %72 = phi i32 [ %63, %69 ], [ %99, %70 ]
  %73 = phi i64 [ 0, %69 ], [ %103, %70 ]
  tail call void @llvm.aie2p.acquire(i32 53, i32 -1)
  %cond4 = icmp eq i32 %72, 1
  %spec.select10 = select i1 %cond4, ptr @stream_a_6_cons_buff_1, ptr @stream_a_6_cons_buff_0
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  %cond5 = icmp eq i32 %71, 1
  %74 = select i1 %cond5, ptr @stream_b_6_cons_buff_1, ptr @stream_b_6_cons_buff_0
  tail call void @f2_step(ptr nonnull %spec.select10, ptr nonnull %74, ptr nonnull @buf_48, ptr nonnull @buf_49, ptr nonnull @buf_50, ptr nonnull @buf_53, ptr nonnull @buf_54, ptr nonnull @buf_51, ptr nonnull @buf_52, ptr nonnull @y1_2_buff_0, ptr nonnull @buf_55, i32 0)
  tail call void @llvm.aie2p.release(i32 52, i32 1)
  %75 = add i32 %72, 1
  %76 = icmp sgt i32 %75, 1
  %77 = select i1 %76, i32 0, i32 %75
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  %78 = add i32 %71, 1
  %79 = icmp sgt i32 %78, 1
  %80 = select i1 %79, i32 0, i32 %78
  tail call void @llvm.aie2p.acquire(i32 53, i32 -1)
  %cond4.1 = icmp eq i32 %77, 1
  %spec.select10.1 = select i1 %cond4.1, ptr @stream_a_6_cons_buff_1, ptr @stream_a_6_cons_buff_0
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  %cond5.1 = icmp eq i32 %80, 1
  %81 = select i1 %cond5.1, ptr @stream_b_6_cons_buff_1, ptr @stream_b_6_cons_buff_0
  tail call void @f2_step(ptr nonnull %spec.select10.1, ptr nonnull %81, ptr nonnull @buf_48, ptr nonnull @buf_49, ptr nonnull @buf_50, ptr nonnull @buf_53, ptr nonnull @buf_54, ptr nonnull @buf_51, ptr nonnull @buf_52, ptr nonnull @y1_2_buff_0, ptr nonnull @buf_55, i32 0)
  tail call void @llvm.aie2p.release(i32 52, i32 1)
  %82 = add i32 %77, 1
  %83 = icmp sgt i32 %82, 1
  %84 = select i1 %83, i32 0, i32 %82
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  %85 = add i32 %80, 1
  %86 = icmp sgt i32 %85, 1
  %87 = select i1 %86, i32 0, i32 %85
  tail call void @llvm.aie2p.acquire(i32 53, i32 -1)
  %cond4.2 = icmp eq i32 %84, 1
  %spec.select10.2 = select i1 %cond4.2, ptr @stream_a_6_cons_buff_1, ptr @stream_a_6_cons_buff_0
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  %cond5.2 = icmp eq i32 %87, 1
  %88 = select i1 %cond5.2, ptr @stream_b_6_cons_buff_1, ptr @stream_b_6_cons_buff_0
  tail call void @f2_step(ptr nonnull %spec.select10.2, ptr nonnull %88, ptr nonnull @buf_48, ptr nonnull @buf_49, ptr nonnull @buf_50, ptr nonnull @buf_53, ptr nonnull @buf_54, ptr nonnull @buf_51, ptr nonnull @buf_52, ptr nonnull @y1_2_buff_0, ptr nonnull @buf_55, i32 0)
  tail call void @llvm.aie2p.release(i32 52, i32 1)
  %89 = add i32 %84, 1
  %90 = icmp sgt i32 %89, 1
  %91 = select i1 %90, i32 0, i32 %89
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  %92 = add i32 %87, 1
  %93 = icmp sgt i32 %92, 1
  %94 = select i1 %93, i32 0, i32 %92
  %95 = or disjoint i64 %73, 3
  tail call void @llvm.aie2p.acquire(i32 53, i32 -1)
  %cond4.3 = icmp eq i32 %91, 1
  %spec.select10.3 = select i1 %cond4.3, ptr @stream_a_6_cons_buff_1, ptr @stream_a_6_cons_buff_0
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  %cond5.3 = icmp eq i32 %94, 1
  %96 = select i1 %cond5.3, ptr @stream_b_6_cons_buff_1, ptr @stream_b_6_cons_buff_0
  tail call void @f2_step(ptr nonnull %spec.select10.3, ptr nonnull %96, ptr nonnull @buf_48, ptr nonnull @buf_49, ptr nonnull @buf_50, ptr nonnull @buf_53, ptr nonnull @buf_54, ptr nonnull @buf_51, ptr nonnull @buf_52, ptr nonnull @y1_2_buff_0, ptr nonnull @buf_55, i32 0)
  tail call void @llvm.aie2p.release(i32 52, i32 1)
  %97 = add i32 %91, 1
  %98 = icmp sgt i32 %97, 1
  %99 = select i1 %98, i32 0, i32 %97
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  %100 = add i32 %94, 1
  %101 = icmp sgt i32 %100, 1
  %102 = select i1 %101, i32 0, i32 %100
  %103 = add nuw nsw i64 %73, 4
  %104 = icmp samesign ult i64 %95, 115
  br i1 %104, label %70, label %105

105:                                              ; preds = %70
  %106 = add nuw nsw i64 %41, 1
  %107 = icmp samesign ult i64 %41, 2
  br i1 %107, label %.preheader, label %108

108:                                              ; preds = %105
  tail call void @llvm.aie2p.acquire(i32 48, i32 -1)
  tail call void @f2_out(ptr nonnull @buf_48, ptr nonnull @y1_2_buff_0, i32 0)
  tail call void @llvm.aie2p.release(i32 49, i32 1)
  tail call void @llvm.aie2p.acquire(i32 48, i32 -1)
  tail call void @f2_out(ptr nonnull @buf_48, ptr nonnull @y1_2_buff_0, i32 1)
  tail call void @llvm.aie2p.release(i32 49, i32 1)
  %109 = add nuw nsw i64 %3, 1
  %.not = icmp eq i64 %109, 9223372036854775807
  br i1 %.not, label %110, label %.preheader11

110:                                              ; preds = %108
  ret void
}

attributes #0 = { mustprogress nocallback nofree nosync nounwind willreturn }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
