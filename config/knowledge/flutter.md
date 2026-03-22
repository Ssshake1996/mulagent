# Flutter & Dart Knowledge Base

## Architecture

### Layer Violations to Detect

| Violation | Symptom | Fix |
|---|---|---|
| Business logic in Widget | `setState` with complex calculations | Extract to Controller/BLoC/Notifier |
| Data model leaks across layers | API response model used in UI | Map to domain entity, then to UI model |
| Cross-layer imports | UI imports data layer directly | Depend on domain abstractions (interfaces) |
| Framework leaks into pure Dart | `import 'package:flutter/...'` in domain | Keep domain layer pure Dart, no Flutter imports |
| Circular dependency | A depends on B, B depends on A | Extract shared interface to common package |
| Direct instantiation in logic | `final repo = ApiRepository()` in BLoC | Use dependency injection (get_it, injectable, riverpod) |

## State Management

### General Rules (library-agnostic)

| Anti-Pattern | Problem | Fix |
|---|---|---|
| Boolean flag soup: `isLoading, hasError, isEmpty` | Impossible states possible | Use sealed class/enum: `Loading, Error(msg), Data(items), Empty` |
| Non-exhaustive state handling | Forgotten states cause blank screens | Pattern match ALL states with `switch`/`when` |
| Subscribing to state in `build()` | Rebuilds on every change | Subscribe in `initState` or use framework selectors |
| Stream/subscription not cancelled | Memory leak | Cancel in `dispose()`, or use framework auto-cleanup |
| Global mutable state | Untraceable bugs, test failures | Use scoped state (Provider/Riverpod scope, BLoC per feature) |

### BLoC/Cubit Pattern
```dart
// GOOD: Sealed states
sealed class UserState {}
class UserLoading extends UserState {}
class UserLoaded extends UserState {
  final User user;
  UserLoaded(this.user);
}
class UserError extends UserState {
  final String message;
  UserError(this.message);
}

// BLoC handles events → emits states
class UserBloc extends Bloc<UserEvent, UserState> {
  UserBloc() : super(UserLoading()) {
    on<LoadUser>((event, emit) async {
      emit(UserLoading());
      try {
        final user = await repo.getUser(event.id);
        emit(UserLoaded(user));
      } catch (e) {
        emit(UserError(e.toString()));
      }
    });
  }
}
```

## Widget Composition

| Anti-Pattern | Problem | Fix |
|---|---|---|
| `build()` > 80 lines | Hard to read, slow rebuilds | Extract sub-widgets as separate classes |
| `_buildSomething()` helper methods | No individual rebuild optimization | Extract to separate `StatelessWidget` |
| Missing `const` constructors | Unnecessary rebuilds | Add `const` to constructors where possible |
| `StatefulWidget` when stateless works | Unnecessary lifecycle overhead | Use `StatelessWidget` + state management |
| Hardcoded colors/spacing | Inconsistent UI, hard to theme | Use `Theme.of(context)` and design tokens |
| Deep widget nesting (>10 levels) | Unreadable, hard to maintain | Flatten with composition, extract sub-widgets |

## Performance

| Anti-Pattern | Problem | Fix |
|---|---|---|
| Expensive work in `build()` | Janky UI, dropped frames | Move to `initState`, use `FutureBuilder`, or state management |
| `MediaQuery.of(context)` in deeply nested widget | Rebuilds entire subtree on resize | Use `MediaQuery.sizeOf(context)` (Flutter 3.10+) or cache at top level |
| Missing `RepaintBoundary` | Unnecessary repaints propagate | Wrap expensive subtrees with `RepaintBoundary` |
| `ListView` without `itemExtent` | Slower scroll performance | Add `itemExtent` when items have fixed height |
| Large images without caching | Repeated network loads, OOM | Use `cached_network_image`, set `cacheWidth`/`cacheHeight` |
| `AnimationController` without `vsync` | Animations run when off-screen | Use `SingleTickerProviderStateMixin` |

## Resource Lifecycle

| Anti-Pattern | Problem | Fix |
|---|---|---|
| Missing `dispose()` | Memory leak, resource leak | Dispose controllers, streams, subscriptions, focus nodes |
| `BuildContext` used after `await` | Widget might be unmounted | Check `mounted` before using context: `if (mounted) setState(...)` |
| `setState` after `dispose` | Runtime error | Guard with `if (mounted)` or cancel async work in `dispose()` |
| `TextEditingController` not disposed | Memory leak | Create in `initState`, dispose in `dispose()` |
| `StreamSubscription` not cancelled | Callbacks on disposed widget | `late StreamSubscription _sub;` → `_sub.cancel()` in `dispose()` |

## Dart Idioms

| Anti-Pattern | Idiomatic Dart |
|---|---|
| `if (x != null) { return x; } else { return y; }` | `return x ?? y;` |
| `list.where((e) => e.isActive).toList().length > 0` | `list.any((e) => e.isActive)` |
| `final List<String> items = [];` | `final items = <String>[];` (type inference) |
| `x == null ? y : x` | `x ?? y` |
| Manual null check + cast | `if (widget is MyWidget) { widget.doThing(); }` |
| `switch` on String for routing | Use `sealed class` or enum with extension methods |
| `print()` for debugging | Use `debugPrint()` (throttled) or `log()` from `dart:developer` |

## Error Handling

```dart
// GOOD: Typed errors with sealed class
sealed class AppError {
  String get message;
}
class NetworkError extends AppError {
  final int statusCode;
  NetworkError(this.statusCode);
  @override String get message => 'Network error: $statusCode';
}
class CacheError extends AppError {
  @override String get message => 'Cache unavailable';
}

// GOOD: Result type pattern
class Result<T> {
  final T? data;
  final AppError? error;
  bool get isSuccess => data != null;
  Result.success(this.data) : error = null;
  Result.failure(this.error) : data = null;
}
```

## Testing

```bash
# Unit tests
flutter test

# With coverage
flutter test --coverage
genhtml coverage/lcov.info -o coverage/html

# Integration tests
flutter test integration_test/

# Analyze
flutter analyze
dart fix --apply
```

## Common Build Errors

| Error | Fix |
|---|---|
| `The argument type 'X' can't be assigned to 'Y'` | Fix type, add explicit cast or conversion |
| `The method 'X' isn't defined` | Check import, package version, spelling |
| `A non-null value must be returned` | Add return statement, or make return type nullable |
| `The parameter 'X' is required` | Add required argument or make parameter optional |
| `Undefined name 'X'` | Add import, check scope |
| `MissingPluginException` | Run `flutter clean && flutter pub get` |
| `Gradle build failed` (Android) | Check `build.gradle`, `compileSdkVersion`, `minSdkVersion` |
| `CocoaPods error` (iOS) | `cd ios && pod install --repo-update` |

## Accessibility

| Check | Requirement |
|---|---|
| `Semantics` label | All interactive widgets must have semantic labels |
| Tap target size | Minimum 48x48dp for touch targets |
| Color contrast | Don't rely on color alone for information |
| Screen reader | Test with TalkBack (Android) and VoiceOver (iOS) |
